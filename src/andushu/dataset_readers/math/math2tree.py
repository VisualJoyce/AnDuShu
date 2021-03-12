import json
import logging
import re
import warnings
from typing import List, Dict

import numpy as np
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import (
    Token,
    Tokenizer,
    SpacyTokenizer,
    PretrainedTransformerTokenizer,
)
from nltk import Tree
from overrides import overrides

# from andushu.dataset_readers.math.equation2tree import equation2tree, callable2tree
from andushu.dataset_readers.math.equation2tree import AstParser, equation2tree, parse_answer, eval_tree, update_tree

logger = logging.getLogger(__name__)


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
            source_tokenizer: Tokenizer = None,
            target_tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = None,
            target_token_indexers: Dict[str, TokenIndexer] = None,
            **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self.ast_parser = AstParser()
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
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

    def pformat_flat(self, tree, nodesep="", parens="()", quotes=False):
        childstrs = []
        for child in tree:
            if isinstance(child, Tree):
                childstrs.append(self.pformat_flat(child, nodesep, parens, quotes))
            elif isinstance(child, tuple):
                childstrs.append("/".join(child))
            elif isinstance(child, str) and not quotes:
                childstrs.append("%s" % child)
            else:
                childstrs.append(repr(child))
        if isinstance(tree._label, str):
            return "%s %s %s %s %s" % (
                parens[0],
                tree._label,
                nodesep,
                " ".join(childstrs),
                parens[1],
            )
        else:
            return "%s %s %s %s %s" % (
                parens[0],
                repr(tree._label),
                nodesep,
                " ".join(childstrs),
                parens[1],
            )

    @overrides
    def _read(self, file_path):
        if 'math23k' in file_path.lower():
            for item in self._read_expression(file_path):
                yield self.text_to_instance(item)
        elif 'mathqa' in file_path.lower():
            for item in self._read_callable(file_path):
                yield self.text_to_instance(item)

    def _read_callable(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            for item in json.load(f):
                flag = True
                for k in ['floor', 'rhombus_perimeter', 'negate_prob',
                          'choose', 'circle_area', 'volume_cone',
                          'circumface', 'original_price_before_gain',
                          'diagonal',
                          'surface_cylinder', 'min', 'tangent', 'sine',
                          'reminder', 'original_price_before_loss', 'p_after_gain',
                          'lcm', 'factorial', 'gcd', 'max', 'surface_sphere', 'volume_sphere',
                          'surface_rectangular_prism',
                          'rhombus_area', 'quadrilateral_area',
                          'volume_cylinder', 'permutation',
                          'square_edge_by_perimeter', 'speed_in_still_water', 'triangle_area_three_edges',
                          'volume_rectangular_prism', 'log']:
                    if k in item['annotated_formula']:
                        flag = False
                        break

                if flag:
                    try:
                        tree = self.ast_parser.parse(item['annotated_formula'])
                    except SyntaxError:
                        # syntax_error.append(item)
                        continue

                    update_tree(tree)

                    tree = self.pformat_flat(tree)

                    options = [item for item in re.findall('[a-e] \) ([^,]*)', item['options'])]
                    item['answer'] = options[ord(item['correct']) - ord('a')]

                    if ':' in item['answer']:
                        continue

                    try:
                        val = eval_tree(tree)
                    except ZeroDivisionError:
                        # zero_division.append(item)
                        continue

                    try:
                        ans = parse_answer(item['answer'])

                        if ans != 'none':
                            err = abs(val - eval(ans))
                            if err < 1e-9:
                                item['ans'] = eval(ans)
                                item['segmented_text'] = item['Problem']
                                item['equation'] = tree
                                yield item

                    except Exception as e:
                        print('---------------')
                        print(e, item)
                        print(item['correct'], options, item['options'])
                        print(val, ans)
                        break

    def _read_expression(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            for item in json.load(f):
                if "千米/小时" in item["equation"]:
                    item["equation"] = item["equation"][:-5]
                # print('{} --------------------------{} {}'.format(item['id'], item['equation'], item['ans']))
                #
                # item['en'] = re.sub('([\d\.]+)times', r'\1 times', item['en'])
                # item['en'] = re.sub('([\d\.]+) %', r'\1%', item['en'])
                # # item['en'] = re.sub('([\d\.]+) \/ ([\d\.]+)', r'\1/\2', item['en'])
                # item['en'] = re.sub('(\d+),(\d+)', r'\1\2', item['en']).replace('%', ' / 100')
                item['ans'] = re.sub('(\d+)\(\(', r'\1+((', item['ans'])
                # tokens_list = ['--DELIMITER--'] + [t.text for t in self._target_token_indexers['tokens'].spacy(text2int(item['en']))]
                try:
                    tree, ans, val, tree_v = equation2tree(self.ast_parser, item['equation'], item['ans'])
                    item['equation'] = self.pformat_flat(tree)
                    yield item
                except:
                    print(item)

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
        source_string = record['segmented_text']
        target_string = record['equation']

        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

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
