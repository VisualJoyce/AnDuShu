import warnings
from inspect import signature
from typing import List, Callable, Tuple, Dict, cast, TypeVar

import torch
from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.nn.beam_search import Sampler, DeterministicSampler

StateType = Dict[str, torch.Tensor]
StepFunctionTypeWithTimestep = Callable[
    [torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]
]
StepFunctionTypeNoTimestep = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]

StepFunctionType = TypeVar(
    "StepFunctionType", StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep
)
"""
The type of step function that can be passed to [`BeamSearch.search`](#search).

This can either be [`StepFunctionTypeWithTimestep`](#stepfunctiontypewithtimestep)
or [`StepFunctionTypeNoTimestep`](#stepfunctiontypenotimestep).
"""


class BeamSearch(FromParams):
    """
    Implements the beam search algorithm for decoding the most likely sequences.

    # Parameters

    end_index : `int`
        The index of the "stop" or "end" token in the target vocabulary.

    max_steps : `int`, optional (default = `50`)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.

    beam_size : `int`, optional (default = `10`)
        The width of the beam used.

    per_node_beam_size : `int`, optional (default = `beam_size`)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See
        [*Beam Search Strategies for Neural Machine Translation*, Freitag and Al-Onaizan, 2017]
        (https://api.semanticscholar.org/CorpusID:2229477).

    sampler : `Sampler`, optional (default = `None`)
        An optional `Sampler` which is used to pick next candidate nodes and beams.
        If not specified, `DeterministicSampler` will be used, which just takes the
        `per_node_beam_size` most likely nodes and the `beam_size` most likely beams.

        Using the [`GumbelSampler`](#gumbelsampler), on the other hand, will give you
        [Stochastic Beam Search](https://api.semanticscholar.org/CorpusID:76662039).
    """

    def __init__(
            self,
            end_index: int,
            nonterminal_index: int,
            # production_start_index: int,
            max_steps: int = 50,
            beam_size: int = 10,
            per_node_beam_size: int = None,
            sampler: Sampler = None,
    ) -> None:
        if not max_steps > 0:
            raise ValueError("max_steps must be positive")
        if not beam_size > 0:
            raise ValueError("beam_size must be positive")
        if per_node_beam_size is not None and not per_node_beam_size > 0:
            raise ValueError("per_node_beam_size must be positive")

        self._end_index = end_index
        self._nonterminal_index = nonterminal_index
        # self._production_start_index = production_start_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.sampler = sampler or DeterministicSampler()

    @staticmethod
    def _reconstruct_sequences(predictions, backpointers):
        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        if not backpointers:
            return reconstructed_predictions

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        return reconstructed_predictions

    # @staticmethod
    # def _reconstruct_sequences(predictions,
    #                            nonterminal_hiddens,
    #                            nonterminal_contexts,
    #                            backpointers):
    #     # Reconstruct the sequences.
    #     # shape: [(batch_size, beam_size, 1)]
    #     reconstructed_predictions = [predictions[-1].unsqueeze(2)]
    #     reconstructed_nonterminal_hiddens = [nonterminal_hiddens[-1]]
    #     reconstructed_nonterminal_contexts = [nonterminal_contexts[-1]]
    #
    #     if not backpointers:
    #         return reconstructed_predictions
    #
    #     # shape: (batch_size, beam_size)
    #     cur_backpointers = backpointers[-1]
    #
    #     batch_size, beam_size, hidden_size = nonterminal_hiddens[0].size()
    #
    #     for timestep in range(len(predictions) - 2, 0, -1):
    #         # shape: (batch_size, beam_size, 1)
    #         cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
    #         cur_nh = nonterminal_hiddens[timestep].gather(1, cur_backpointers.unsqueeze(-1).repeat(1, 1, hidden_size))
    #         cur_nc = nonterminal_contexts[timestep].gather(1, cur_backpointers.unsqueeze(-1).repeat(1, 1, hidden_size))
    #
    #         reconstructed_predictions.append(cur_preds)
    #         reconstructed_nonterminal_hiddens.append(cur_nh)
    #         reconstructed_nonterminal_contexts.append(cur_nc)
    #
    #         # shape: (batch_size, beam_size)
    #         cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)
    #
    #     # shape: (batch_size, beam_size, 1)
    #     final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
    #     final_nh = nonterminal_hiddens[0].gather(1, cur_backpointers.unsqueeze(-1).repeat(1, 1, hidden_size))
    #     final_nc = nonterminal_contexts[0].gather(1, cur_backpointers.unsqueeze(-1).repeat(1, 1, hidden_size))
    #
    #     reconstructed_predictions.append(final_preds)
    #     reconstructed_nonterminal_hiddens.append(final_nh)
    #     reconstructed_nonterminal_contexts.append(final_nc)
    #
    #     return {
    #         'predictions': reconstructed_predictions,
    #         'nonterminal_hiddens': reconstructed_nonterminal_hiddens,
    #         'nonterminal_contexts': reconstructed_nonterminal_contexts,
    #     }

    @torch.no_grad()
    def search(
            self,
            start_predictions: torch.Tensor,
            start_state: StateType,
            step: StepFunctionType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.

        # Notes

        If your step function returns `-inf` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have `-inf` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from `search`
        and potentially discard sequences with non-finite log probability.

        # Parameters

        start_predictions : `torch.Tensor`
            A tensor containing the initial predictions with shape `(batch_size,)`.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.

        start_state : `StateType`
            The initial state passed to the `step` function. Each value of the state dict
            should be a tensor of shape `(batch_size, *)`, where `*` means any other
            number of dimensions.

        step : `StepFunctionType`
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two or three arguments:

            - a tensor of shape `(group_size,)` representing the index of the predicted
            tokens from the last time step,
            - the current state, a `StateType`, and
            - optionally, the timestep, an `int`.

            The `group_size` will be `batch_size * beam_size`, except in the initial
            step, for which it will just be `batch_size`.

            The function is expected to return a tuple, where the first element
            is a tensor of shape `(group_size, target_vocab_size)` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            `(group_size, *)`, where `*` means any other number of dimensions.

        # Returns

        `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of `(predictions, log_probabilities)`, where `predictions`
            has shape `(batch_size, beam_size, max_steps)` and `log_probabilities`
            has shape `(batch_size, beam_size)`.
        """
        step_signature = signature(step)
        if len(step_signature.parameters) < 3:
            # If the step function we're given does not take the time step argument, wrap it
            # in one that does.
            old_step = cast(StepFunctionTypeNoTimestep, step)

            def new_step(
                    last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], time_step: int
            ):
                return old_step(last_predictions, state)

            return self._search(start_predictions, start_state, new_step)
        else:
            return self._search(
                start_predictions, start_state, cast(StepFunctionTypeWithTimestep, step)
            )

    def _search(
            self,
            start_predictions: torch.Tensor,
            start_state: StateType,
            step: StepFunctionTypeWithTimestep,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = start_predictions.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        nonterminal_hiddens = []
        nonterminal_contexts = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(start_predictions, start_state, 0)

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ConfigurationError(
                f"Target vocab size ({num_classes:d}) too small "
                f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                f"Please decrease beam_size or per_node_beam_size."
            )

        sampler_state = self.sampler.init_state(
            start_class_log_probabilities, batch_size, num_classes
        )

        # Get the initial predicted classed and their log probabilities.
        # shape: (batch_size, beam_size), (batch_size, beam_size)
        (
            start_top_log_probabilities,
            start_predicted_classes,
            sampler_state,
        ) = self.sampler.sample_beams(start_class_log_probabilities, self.beam_size, sampler_state)

        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn(
                "Empty sequences predicted. You may want to increase the beam size or ensure "
                "your step function is working properly.",
                RuntimeWarning,
            )
            return start_predicted_classes.unsqueeze(
                -1), start_top_log_probabilities, state['decoder_hidden'], state['decoder_context']

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        # Set the same state for each element in the beam.
        self._update_initial_state(state, batch_size)

        # cond_tensor = torch.where(last_predictions == self._nonterminal_index)
        nonterminal_hiddens.append(state['decoder_hidden'].reshape(
            batch_size, self.beam_size, -1
        ))
        nonterminal_contexts.append(state['decoder_context'].reshape(
            batch_size, self.beam_size, -1
        ))

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state, timestep + 1)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities,
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes, sampler_state = self.sampler.sample_nodes(
                cleaned_log_probabilities, self.per_node_beam_size, sampler_state
            )

            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probabilities = (
                last_log_probabilities.unsqueeze(2)
                    .expand(batch_size, self.beam_size, self.per_node_beam_size)
                    .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )

            # Keep only the top `beam_size` beam indices.
            # shape (both): (batch_size, beam_size)
            (
                restricted_beam_log_probs,
                restricted_beam_indices,
                sampler_state,
            ) = self.sampler.sample_beams(reshaped_summed, self.beam_size, sampler_state)

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices
            )

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices // self.per_node_beam_size
            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            self._update_state(state, backpointer)

            # cond_tensor = torch.where(last_predictions == self._nonterminal_index)
            nonterminal_hiddens.append(state['decoder_hidden'].reshape(
                batch_size, self.beam_size, -1
            ))
            nonterminal_contexts.append(state['decoder_context'].reshape(
                batch_size, self.beam_size, -1
            ))

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn(
                "Infinite log probabilities encountered. Some final sequences may not make sense. "
                "This can happen when the beam size is larger than the number of valid (non-zero "
                "probability) transitions that the step function produces.",
                RuntimeWarning,
            )

        reconstructed_predictions = self._reconstruct_sequences(predictions, backpointers)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        # print(predictions, reconstructed_predictions, all_predictions)

        # reconstructed = self._reconstruct_sequences(predictions,
        #                                             nonterminal_hidden,
        #                                             nonterminal_context,
        #                                             backpointers)
        #
        # # shape: (batch_size, beam_size, max_steps)
        # all_predictions = torch.cat(list(reversed(reconstructed['predictions'])), 2)
        # nonterminal_hiddens = list(reversed(reconstructed['nonterminal_hiddens']))
        # nonterminal_contexts = list(reversed(reconstructed['nonterminal_contexts']))

        return all_predictions, last_log_probabilities, nonterminal_hiddens, nonterminal_contexts

    @staticmethod
    def _is_multilayer_rnn_decoder(key: str, state_tensor: torch.Tensor) -> bool:
        return state_tensor.dim() == 3 and key in {
            "decoder_hidden",
            "decoder_context",
        }

    def _update_initial_state(self, state: StateType, batch_size: int):
        """
        Expand tensors in a state dictionary from `(batch_size, *)` to `(batch_size * beam_size, *)`.
        """
        for key, state_tensor in state.items():
            if state_tensor is None:
                continue
            multilayer_rnn_decoder = self._is_multilayer_rnn_decoder(key, state_tensor)

            if multilayer_rnn_decoder:
                # shape: (num_layers, batch_size * beam_size, *)
                num_layers, _, *last_dims = state_tensor.size()
                state[key] = (
                    state_tensor.unsqueeze(2)
                        .expand(num_layers, batch_size, self.beam_size, *last_dims)
                        .reshape(num_layers, batch_size * self.beam_size, *last_dims)
                )
            else:
                # shape: (batch_size * beam_size, *)
                _, *last_dims = state_tensor.size()
                state[key] = (
                    state_tensor.unsqueeze(1)
                        .expand(batch_size, self.beam_size, *last_dims)
                        .reshape(batch_size * self.beam_size, *last_dims)
                )

    def _update_state(self, state: StateType, backpointer: torch.Tensor):
        batch_size = backpointer.size()[0]

        for key, state_tensor in state.items():
            if state_tensor is None:
                continue
            multilayer_rnn_decoder = self._is_multilayer_rnn_decoder(key, state_tensor)

            if multilayer_rnn_decoder:
                # shape: (num_layers, batch_size * beam_size, *)
                num_layers, _, *last_dims = state_tensor.size()
                expanded_backpointer = backpointer.view(
                    batch_size, self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, self.beam_size, *last_dims)
                expanded_backpointer = expanded_backpointer.unsqueeze(0).repeat(num_layers, 1, 1, 1)
                # shape: (num_layers, batch_size * beam_size, *)
                state[key] = (
                    state_tensor.reshape(num_layers, batch_size, self.beam_size, *last_dims)
                        .gather(2, expanded_backpointer)
                        .reshape(num_layers, batch_size * self.beam_size, *last_dims)
                )
            else:
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.view(
                    batch_size, self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, self.beam_size, *last_dims)
                # shape: (batch_size * beam_size, *)
                state[key] = (
                    state_tensor.reshape(batch_size, self.beam_size, *last_dims)
                        .gather(1, expanded_backpointer)
                        .reshape(batch_size * self.beam_size, *last_dims)
                )
