from collections import defaultdict
from typing import Dict, List, Optional, NamedTuple, Tuple

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import TokenIndexer, Token, IndexedTokenList
from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util
from nltk import Production, Nonterminal
from overrides import overrides


class ProductionRule(NamedTuple):
    rule: Production
    # is_global_rule: bool
    rule_id: Optional[torch.LongTensor] = None
    nonterminal: Optional[str] = None


# This is just here for backward compatability.
ProductionRuleArray = ProductionRule

ProductionRuleFieldTensors = Dict[str, Dict[str, torch.Tensor]]


# mypy doesn't like that we're using a crazy data type - the data type we use here is _supposed_ to
# be in the bounds of DataArray, but ProductionRule definitely isn't.  TODO(mattg): maybe we
# should find a better way to loosen those bounds, or let people extend them.  E.g., we could have
# DataArray be a class, and let people subclass it, or something.
class ProductionRuleField(Field[ProductionRule]):  # type: ignore
    """
    This ``Field`` represents a production rule from a grammar, like "S -> [NP, VP]", "N -> John",
    or "<b,c> -> [<a,<b,c>>, a]".

    We assume a few things about how these rules are formatted:

        - There is a left-hand side (LHS) and a right-hand side (RHS), where the LHS is always a
          non-terminal, and the RHS is either a terminal, a non-terminal, or a sequence of
          terminals and/or non-terminals.
        - The LHS and the RHS are joined by " -> ", and this sequence of characters appears nowhere
          else in the rule.
        - Non-terminal sequences in the RHS are formatted as "[NT1, NT2, ...]".
        - Some rules come from a global grammar used for a whole dataset, while other rules are
          specific to a particular ``Instance``.

    We don't make use of most of these assumptions in this class, but the code that consumes this
    ``Field`` relies heavily on them in some places.

    If the given rule is in the global grammar, we treat the rule as a vocabulary item that will
    get an index and (in the model) an embedding.  If the rule is not in the global grammar, we do
    not create a vocabulary item from the rule, and don't produce a tensor for the rule - we assume
    the model will handle representing this rule in some other way.

    Because we represent global grammar rules and instance-specific rules differently, this
    ``Field`` does not lend itself well to batching its arrays, even in a sequence for a single
    training instance.  A model using this field will have to manually batch together rule
    representations after splitting apart the global rules from the ``Instance`` rules.

    In a model, this will get represented as a ``ProductionRule``, which is defined above.
    This is a namedtuple of ``(rule_string, is_global_rule, [rule_id], nonterminal)``, where the
    ``rule_id`` ``Tensor``, if present, will have shape ``(1,)``.  We don't do any batching of the
    ``Tensors``, so this gets passed to ``Model.forward()`` as a ``List[ProductionRule]``.  We
    pass along the rule string because there isn't another way to recover it for instance-specific
    rules that do not make it into the vocabulary.

    Parameters
    ----------
    rule : ``str``
        The production rule, formatted as described above.  If this field is just padding, ``rule``
        will be the empty string.
    is_global_rule : ``bool``
        Whether this rule comes from the global grammar or is an instance-specific production rule.
    vocab_namespace : ``str``, optional (default="rule_labels")
        The vocabulary namespace to use for the global production rules.  We use "rule_labels" by
        default, because we typically do not want padding and OOV tokens for these, and ending the
        namespace with "labels" means we don't get padding and OOV tokens.
    nonterminal : ``str``, optional, default = None
        The left hand side of the rule. Sometimes having this as separate part of the ``ProductionRule``
        can deduplicate work.
    """

    def __init__(
            self,
            rule: Production,
            # is_global_rule: bool,
            tokens: List[Token],
            parent_non_terminal_index: Tuple,
            token_indexers: Optional[Dict[str, TokenIndexer]] = None,
            vocab_namespace: str = "rule_labels",
            nonterminal: str = None,
    ) -> None:
        self.rule = rule
        self.nonterminal = nonterminal
        # self.is_global_rule = is_global_rule
        self._vocab_namespace = vocab_namespace
        # self._rule_id: int = None
        self._parent_non_terminal_index = parent_non_terminal_index

        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens: Optional[Dict[str, IndexedTokenList]] = None

    @property
    def token_indexers(self) -> Dict[str, TokenIndexer]:
        if self._token_indexers is None:
            raise ValueError(
                "TextField's token_indexers have not been set.\n"
                "Did you forget to call DatasetReader.apply_token_indexers(instance) "
                "on your instance?\n"
                "If apply_token_indexers() is being called but "
                "you're still seeing this error, it may not be implemented correctly."
            )
        return self._token_indexers

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self.token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)
        # if self.is_global_rule:
        #     counter[self._vocab_namespace][self.rule] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        # if self.is_global_rule and self._rule_id is None:
        #     self._rule_id = vocab.get_token_index(self.rule, self._vocab_namespace)

        self._indexed_tokens = {}
        for indexer_name, indexer in self.token_indexers.items():
            self._indexed_tokens[indexer_name] = indexer.tokens_to_indices(self.tokens, vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        The `TextField` has a list of `Tokens`, and each `Token` gets converted into arrays by
        (potentially) several `TokenIndexers`.  This method gets the max length (over tokens)
        associated with each of these arrays.
        """
        if self._indexed_tokens is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before determining padding lengths."
            )

        padding_lengths = {}
        for indexer_name, indexer in self.token_indexers.items():
            indexer_lengths = indexer.get_padding_lengths(self._indexed_tokens[indexer_name])
            for key, length in indexer_lengths.items():
                padding_lengths[f"{indexer_name}___{key}"] = length
        return padding_lengths

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> ProductionRuleFieldTensors:
        if self._indexed_tokens is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before calling .as_tensor()"
            )

        tensors = {
            'parent_non_terminal_index': torch.LongTensor(self._parent_non_terminal_index)
        }

        indexer_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        for key, value in padding_lengths.items():
            # We want this to crash if the split fails. Should never happen, so I'm not
            # putting in a check, but if you fail on this line, open a github issue.
            indexer_name, padding_key = key.split("___")
            indexer_lengths[indexer_name][padding_key] = value

        for indexer_name, indexer in self.token_indexers.items():
            tensors[indexer_name] = indexer.as_padded_tensor_dict(
                self._indexed_tokens[indexer_name], indexer_lengths[indexer_name]
            )
        return tensors
        # if self.is_global_rule:
        #     tensor = torch.LongTensor([self._rule_id])
        # else:
        #     tensor = None
        # return ProductionRule(self.rule, self.is_global_rule, tensor, self.nonterminal)

    @overrides
    def empty_field(self):
        text_field = ProductionRuleField(Production(Nonterminal(''), ()), [], (0, 0), self._token_indexers)
        text_field._indexed_tokens = {}
        if self._token_indexers is not None:
            for indexer_name, indexer in self.token_indexers.items():
                text_field._indexed_tokens[indexer_name] = indexer.get_empty_token_list()
        return text_field
        # # This _does_ get called, because we don't want to bother with modifying the ListField to
        # # ignore padding for these.  We just make sure the rule is the empty string, which the
        # # model will use to know that this rule is just padding.
        # return ProductionRuleField(rule="", is_global_rule=False)

    @overrides
    def batch_tensors(
            self, tensor_list: List[ProductionRuleFieldTensors]
    ) -> ProductionRuleFieldTensors:  # type: ignore
        # This is creating a dict of {token_indexer_name: {token_indexer_outputs: batched_tensor}}
        # for each token indexer used to index this field.
        parent_non_terminal_index = [d.pop('parent_non_terminal_index') for d in tensor_list]

        indexer_lists: Dict[str, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        for tensor_dict in tensor_list:
            for indexer_name, indexer_output in tensor_dict.items():
                indexer_lists[indexer_name].append(indexer_output)
        batched_tensors = {
            # NOTE(mattg): if an indexer has its own nested structure, rather than one tensor per
            # argument, then this will break.  If that ever happens, we should move this to an
            # `indexer.batch_tensors` method, with this logic as the default implementation in the
            # base class.
            indexer_name: util.batch_tensor_dicts(indexer_outputs)
            for indexer_name, indexer_outputs in indexer_lists.items()
        }
        batched_tensors['parent_non_terminal_index'] = torch.stack(parent_non_terminal_index)
        return batched_tensors

    def __str__(self) -> str:
        return (
            f"ProductionRuleField with rule: {self.rule} (is_global_rule: "
            f"{self.is_global_rule}) in namespace: '{self._vocab_namespace}'.'"
        )
