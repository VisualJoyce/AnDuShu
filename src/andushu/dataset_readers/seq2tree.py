import logging
from typing import Dict, List

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from nltk import Tree, Production, Nonterminal
from nltk.tree.tree import _child_names

from andushu.data.fields import ProductionRuleField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

NONTERMINAL_SYMBOL = "@non@"
PRODUCTION_START_SYMBOL = "@(@"


def subtree_stub(tree):
    nodes = []
    for p in tree.treepositions():
        if len(p) > 1:
            if [p[:1], 'keep'] in nodes:
                nodes.remove([p[:1], 'keep'])
                nodes.append([p[:1], 'replace'])
        else:
            nodes.append([p, 'keep'])

    tokens = []
    for p, t in nodes:
        if t == 'replace':
            token = NONTERMINAL_SYMBOL
        else:
            token = tree[p].label() if isinstance(tree[p], Tree) else tree[p]
        tokens.append(token)
    return tokens


def productions(target):
    tree = Tree.fromstring(target)

    prods = [((0, 0), Production(Nonterminal(tree._label), _child_names(tree)), subtree_stub(tree))]
    prods_index = {(): 0}
    for i, child in enumerate(tree):
        if isinstance(child, Tree):
            childpos = child.treepositions()
            positions = ((i,) + p for p in childpos)
        else:
            positions = ((i,))

        for p in positions:
            child = tree[p]
            if isinstance(child, Tree):
                prods_index[p] = len(prods_index)
                prods.append(((prods_index[p[:-1]], p[-1] + 1),
                              Production(Nonterminal(child._label), _child_names(child)), subtree_stub(child)))
    return prods


@DatasetReader.register("seq2tree")
class Seq2TreeDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.
    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True) -> None:
        super().__init__()
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            example_iter = (line.strip("\n") for line in data_file if line.strip("\n"))
            filtered_example_iter = (
                example for example in example_iter
            )
            for line in self.shard_iterable(filtered_example_iter):
                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line: %s" % line)
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)

    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {"source_tokens": source_field}
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))

            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields_dict["target_tokens"] = target_field

            production_rule_fields: List[Field] = []
            for parent_non_terminal_index, production, tokens in productions(target_string):
                if parent_non_terminal_index == (0, 0):
                    start_token = START_SYMBOL
                else:
                    start_token = PRODUCTION_START_SYMBOL

                x, y = parent_non_terminal_index

                field = ProductionRuleField(production,
                                            [Token(start_token)] + [Token(t) for t in tokens] + [Token(END_SYMBOL)],
                                            (x, y),
                                            token_indexers=self._target_token_indexers)
                production_rule_fields.append(field)

            fields_dict['production_rules'] = ListField(production_rule_fields)
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
