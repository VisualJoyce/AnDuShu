import json
import os
import re
import warnings
from typing import List, Dict

import jieba
import jsonlines
import numpy as np
from allennlp.common.util import START_SYMBOL, END_SYMBOL, logger
from allennlp.data import Tokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import (
    Token,
    PretrainedTransformerTokenizer, SpacyTokenizer,
)
from overrides import overrides

from andushu.dataset_readers.math.equation2tree import AstParser, equation2tree, parse_answer, eval_tree, update_tree, \
    pformat_flat, isfloat, filtered_ops


class Processor:

    @classmethod
    def process_math23k(cls, ast_parser, item, use_chinese_segmentation):
        if "千米/小时" in item["equation"]:
            item["equation"] = item["equation"][:-5]
        item['ans'] = re.sub('(\d+)\(\(', r'\1+((', item['ans'])
        lang = item.get('lang', 'zh')
        try:
            tree, ans, val, tree_v = equation2tree(ast_parser, item['equation'], item['ans'])
            item['equation'] = pformat_flat(tree)
            if lang == 'zh':
                tokens = []
                for w in jieba.cut(''.join(item['segmented_text'].split())):
                    try:
                        w_val = eval(w)
                        w_int = int(w_val)
                        if w_int == w_val:
                            tokens.append(str(w_int))
                        else:
                            tokens.append(w)
                    except:
                        tokens.append(w)

                item['segmented_text'] = ' '.join(re.split('(\d+.?\d+)', ' '.join(tokens)))
                if not use_chinese_segmentation:
                    tl = []
                    for w in item['segmented_text'].split():
                        if isfloat(w):
                            tl.append(w)
                        else:
                            tl.extend(list(w))
                    item['segmented_text'] = ' '.join(tl)
            item['problem'] = item['segmented_text']
            item['process_type'] = 'math23k'
            return item
        except Exception as e:
            print(e, item)

    @classmethod
    def process_mathqa(cls, ast_parser, item, ops):
        if len(item['Problem'].split()) > 128:
            # print(item)
            raise ValueError('Problem too long!')

        if all([k not in item['annotated_formula'] for k in ops]):
            tree = ast_parser.parse(item['annotated_formula'])
            tree = update_tree(tree)
            tree = pformat_flat(tree)

            options = [item for item in re.findall('[a-e] \) ([^,]*)', item['options'])]
            item['answer'] = options[ord(item['correct']) - ord('a')]

            if ':' in item['answer']:
                raise ValueError('Answer is not acceptable!')

            val = eval_tree(tree)

            answer = parse_answer(item['answer'])
            decimal_len = answer[::-1].find('.')

            ans = eval(answer)
            if decimal_len > 0:
                err = abs(round(val, decimal_len) - ans)
            else:
                err = abs(val - ans)

            if ans != 'none' and abs(err / val) < 1e-4:
                item['ans'] = ans
                item['problem'] = item['Problem']
                item['equation'] = tree
                item['process_type'] = 'mathqa'
                return item
            else:
                raise ValueError('Answer is not acceptable!')


@DatasetReader.register("copynet_math2tree")
class Math2TreeDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    `CopyNet` model, or any model with a matching API.

    The expected format for each input line is: <source_sequence_string><tab><target_sequence_string>.
    An instance produced by `CopyNetDatasetReader` will containing at least the following fields:

    - `source_tokens`: a `TextField` containing the tokenized source sentence.
       This will result in a tensor of shape `(batch_size, source_length)`.

    - `source_token_ids`: an `ArrayField` of size `(batch_size, source_length)`
      that contains an ID for each token in the source sentence. Tokens that
      match at the lowercase level will share the same ID. If `target_tokens`
      is passed as well, these IDs will also correspond to the `target_token_ids`
      field, i.e. any tokens that match at the lowercase level in both
      the source and target sentences will share the same ID. Note that these IDs
      have no correlation with the token indices from the corresponding
      vocabulary namespaces.

    - `source_to_target`: a `NamespaceSwappingField` that keeps track of the index
      of the target token that matches each token in the source sentence.
      When there is no matching target token, the OOV index is used.
      This will result in a tensor of shape `(batch_size, source_length)`.

    - `metadata`: a `MetadataField` which contains the source tokens and
      potentially target tokens as lists of strings.

    When `target_string` is passed, the instance will also contain these fields:

    - `target_tokens`: a `TextField` containing the tokenized target sentence,
      including the `START_SYMBOL` and `END_SYMBOL`. This will result in
      a tensor of shape `(batch_size, target_length)`.

    - `target_token_ids`: an `ArrayField` of size `(batch_size, target_length)`.
      This is calculated in the same way as `source_token_ids`.

    See the "Notes" section below for a description of how these fields are used.

    # Parameters

    target_namespace : `str`, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the NamespaceSwappingField.
    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.

    # Notes

    In regards to the fields in an `Instance` produced by this dataset reader,
    `source_token_ids` and `target_token_ids` are primarily used during training
    to determine whether a target token is copied from a source token (or multiple matching
    source tokens), while `source_to_target` is primarily used during prediction
    to combine the copy scores of source tokens with the generation scores for matching
    tokens in the target namespace.
    """

    def __init__(
            self,
            read_type: str,
            op_type: str,
            source_tokenizer: Tokenizer = None,
            target_tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = None,
            target_token_indexers: Dict[str, TokenIndexer] = None,
            **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        assert read_type in ['math23k', 'mathqa', 'mathxling']
        self.read_type = read_type
        self.op_type = op_type
        self.ast_parser = AstParser()
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._chinese_segmentation = True if self._source_tokenizer.spacy.lang == 'zh' else False
        if (
                isinstance(self._target_tokenizer, PretrainedTransformerTokenizer)
                and self._target_tokenizer._add_special_tokens
        ):
            warnings.warn(
                "'add_special_tokens' is True for target_tokenizer, which is a PretrainedTransformerTokenizer. "
                "This means special tokens, such as '[CLS]' and '[SEP]', will probably end up in "
                "your model's predicted target sequences. "
                "If this is not what you intended, make sure to specify 'add_special_tokens: False' for "
                "your target_tokenizer.",
                UserWarning,
            )

    @overrides
    def _read(self, file_path):
        func = getattr(self, f'_read_{self.read_type}')
        for item in self.shard_iterable(func(file_path, op_type=self.op_type)):
            yield self.text_to_instance(item)

    def _read_and_dump(self, filename, file_path, op_type):
        with jsonlines.open(filename, mode="w") as writer:
            with open(file_path, encoding="utf-8") as f:
                for item in json.load(f):
                    try:
                        if item['process_type'] == 'mathqa':
                            item = Processor.process_mathqa(self.ast_parser, item, filtered_ops[op_type])
                            status = 'ok' if item is not None else 'err'
                        else:
                            item = Processor.process_math23k(self.ast_parser, item,
                                                             use_chinese_segmentation=self._chinese_segmentation)
                            if op_type == "disallow_pow" and item and "Pow" in item['equation']:
                                status = 'err'
                            else:
                                status = 'ok' if item is not None else 'err'
                    except SyntaxError:
                        status = 'err'
                    except ZeroDivisionError:
                        status = 'err'
                    except ValueError:
                        status = 'err'
                    except Exception as e:
                        status = 'err'

                    if status == 'ok':
                        item['process_type'] = 'mathqa'
                        writer.write(item)

                    yield status, item

    def _read_data(self, file_path, op_type):
        errors = []
        total = 0
        filename = file_path + f'.{op_type}.jsonl'
        if os.path.isfile(filename):
            with jsonlines.open(filename) as reader:
                for item in reader:
                    total += 1
                    yield item
        else:
            for status, item in self._read_and_dump(filename, file_path, op_type):
                total += 1
                if status == 'ok':
                    yield item
                else:
                    errors.append(item)
        logger.info(f"Total instances: {total} \n"
                    f"Error instances: {len(errors)} \n"
                    f"Loaded instances: {total - len(errors)}")

    def _read_math23k(self, file_path, op_type):
        return iter(self._read_data(file_path, op_type))

    def _read_mathqa(self, file_path, op_type):
        return iter(self._read_data(file_path, op_type))

    def _read_mathxling(self, file_path, op_type):
        total = 0
        errors = []
        for fp in file_path.split(";"):
            with jsonlines.open(fp) as reader:
                for item in reader:
                    total += 1
                    yield item
        logger.info(f"Total instances: {total} \n"
                    f"Error instances: {len(errors)} \n"
                    f"Loaded instances: {total - len(errors)}")

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text, len(ids)))
        return out

    @overrides
    def text_to_instance(self, record: Dict) -> Instance:  # type: ignore
        """
        Turn raw source string and target string into an `Instance`.

        # Parameters

        source_string : `str`, required
        target_string : `str`, optional (default = `None`)

        # Returns

        `Instance`
            See the above for a description of the fields that the instance will contain.
        """
        source_string = record['problem']
        target_string = record['equation']

        tokenized_source = []
        for t in source_string.split():
            if isfloat(t):
                tokenized_source.append(Token(t))
            else:
                tokenized_source.extend(self._source_tokenizer.tokenize(t))

        if not tokenized_source:
            # If the tokenized source is empty, it will cause issues downstream.
            raise ValueError(f"source tokenizer produced no tokens from source '{source_string}'")

        source_field = TextField(tokenized_source)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source,
                                                        self._target_token_indexers['tokens'].namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source]}
        fields_dict = {"source_tokens": source_field, "source_to_target": source_to_target_field}

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source + tokenized_target)
            source_token_ids = source_and_target_token_ids[: len(tokenized_source)]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source):]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))

            meta_fields['ans'] = record['ans']
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source)
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore
