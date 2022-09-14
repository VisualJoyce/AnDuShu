from typing import Dict, List, Tuple, Optional, Union, Any

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric
from overrides import overrides
from torch.nn import Linear

from .seq_decoder import SeqDecoder
from ..decoder_nets.decoder_net import DecoderNet


@SeqDecoder.register("auto_regressive_seq_decoder")
class AutoRegressiveSeqDecoder(SeqDecoder):
    """
    An autoregressive decoder that can be used for most seq2seq tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : `DecoderNet`, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : `int`, required
        Maximum length of decoded sequences.
    target_embedder : `Embedding`
        Embedder for target tokens.
    target_namespace : `str`, optional (default = `'tokens'`)
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : `int`, optional (default = `4`)
        Width of the beam for beam search.
    tensor_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : `float` optional (default = `0.0`)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            decoder_net: DecoderNet,
            max_decoding_steps: int,
            target_embedder: Embedding,
            target_namespace: str = "tokens",
            tie_output_embedding: bool = False,
            scheduled_sampling_ratio: float = 0,
            label_smoothing_ratio: Optional[float] = None,
            beam_size: int = 4,
            tensor_based_metric: Metric = None,
            token_based_metric: Metric = None,
    ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._decoder_net = decoder_net
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        target_vocab_size = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            self._decoder_net.get_output_dim(), target_vocab_size
        )

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        # These metrics will be updated during training and validation
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric

        self._scheduled_sampling_ratio = scheduled_sampling_ratio

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the beam search, does beam search and returns beam search results.
        """
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _forward_loss(
            self, state: Dict[str, torch.Tensor], target_tokens: TextFieldTensors
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (batch_size, max_target_sequence_length)
        targets = util.get_token_ids_from_text_field_tensors(target_tokens)

        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self.target_embedder(targets)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        if self._scheduled_sampling_ratio == 0 and self._decoder_net.decodes_parallel:
            _, decoder_output = self._decoder_net(
                previous_state=state,
                previous_steps_predictions=target_embedding[:, :-1, :],
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
                previous_steps_mask=target_mask[:, :-1],
            )

            # shape: (group_size, max_target_sequence_length, num_classes)
            logits = self._output_projection_layer(decoder_output)
        else:
            batch_size = source_mask.size()[0]
            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1

            # Initialize target predictions with the start index.
            # shape: (batch_size,)
            last_predictions = source_mask.new_full(
                (batch_size,), fill_value=self._start_index, dtype=torch.long
            )

            # shape: (steps, batch_size, target_embedding_dim)
            steps_embeddings = torch.Tensor([])

            step_logits: List[torch.Tensor] = []

            for timestep in range(num_decoding_steps):
                if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                    # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                    # during training.
                    # shape: (batch_size, steps, target_embedding_dim)
                    state["previous_steps_predictions"] = steps_embeddings

                    # shape: (batch_size, )
                    effective_last_prediction = last_predictions
                else:
                    # shape: (batch_size, )
                    effective_last_prediction = targets[:, timestep]

                    if timestep == 0:
                        state["previous_steps_predictions"] = torch.Tensor([])
                    else:
                        # shape: (batch_size, steps, target_embedding_dim)
                        state["previous_steps_predictions"] = target_embedding[:, :timestep]

                # shape: (batch_size, num_classes)
                output_projections, state = self._prepare_output_projections(
                    effective_last_prediction, state
                )

                # list of tensors, shape: (batch_size, 1, num_classes)
                step_logits.append(output_projections.unsqueeze(1))

                # shape (predicted_classes): (batch_size,)
                _, predicted_classes = torch.max(output_projections, 1)

                # shape (predicted_classes): (batch_size,)
                last_predictions = predicted_classes

                # shape: (batch_size, 1, target_embedding_dim)
                last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

                # This step is required, since we want to keep up two different prediction history: gold and real
                if steps_embeddings.shape[-1] == 0:
                    # There is no previous steps, except for start vectors in `last_predictions`
                    # shape: (group_size, 1, target_embedding_dim)
                    steps_embeddings = last_predictions_embeddings
                else:
                    # shape: (group_size, steps_count, target_embedding_dim)
                    steps_embeddings = torch.cat([steps_embeddings, last_predictions_embeddings], 1)

            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

        # Compute loss.
        target_mask = util.get_text_field_mask(target_tokens)
        loss = self._get_loss(logits, targets, target_mask)

        # TODO: We will be using beam search to get predictions for validation, but if beam size in 1
        # we could consider taking the last_predictions here and building step_predictions
        # and use that instead of running beam search again, if performance in validation is taking a hit
        output_dict = {"loss": loss}

        return output_dict

    def _prepare_output_projections(
            self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoder_state, decoder_output = self._decoder_net(
            previous_state=state,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_predictions=previous_steps_predictions,
        )
        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        if self._decoder_net.decodes_parallel:
            decoder_output = decoder_output[:, -1, :]

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state

    def _get_loss(
            self, logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio
        )

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def take_step(
            self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        # Parameters

        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.
        step : `int`
            The time step in beam search decoding.

        # Returns

        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(
                    self._tensor_based_metric.get_metric(reset=reset)  # type: ignore
                )
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    def forward(
            self,
            encoder_out: Dict[str, torch.LongTensor],
            target_tokens: TextFieldTensors = None,
            metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        state = encoder_out
        decoder_init_state = self._decoder_net.init_decoder_state(state)
        state.update(decoder_init_state)

        if target_tokens:
            state_forward_loss = (
                state if self.training else {k: v.clone() for k, v in state.items() if v is not None}
            )
            output_dict = self._forward_loss(state_forward_loss, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if target_tokens:
                targets = util.get_token_ids_from_text_field_tensors(target_tokens)
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]

                    self._tensor_based_metric(best_predictions, targets)  # type: ignore

                if self._token_based_metric is not None:
                    output_dict = self.post_process(output_dict)
                    predicted_tokens = output_dict["predicted_tokens"]

                    self._token_based_metric(  # type: ignore
                        predicted_tokens,
                        metadata
                    )

        return output_dict

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self.indices_to_tokens(predicted_indices)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def indices_to_tokens(self, batch_indeces: numpy.ndarray) -> List[List[str]]:

        if not isinstance(batch_indeces, numpy.ndarray):
            batch_indeces = batch_indeces.detach().cpu().numpy()

        all_tokens = []
        for indices in batch_indeces:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            tokens = [
                self._vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_tokens.append(tokens)

        return all_tokens


@SeqDecoder.register("auto_regressive_seq_decoder_with_copy")
class AutoRegressiveSeqDecoderWithCopy(SeqDecoder):
    """
    An autoregressive decoder that can be used for most seq2seq tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : `DecoderNet`, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : `int`, required
        Maximum length of decoded sequences.
    target_embedder : `Embedding`
        Embedder for target tokens.
    target_namespace : `str`, optional (default = `'tokens'`)
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : `int`, optional (default = `4`)
        Width of the beam for beam search.
    tensor_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : `float` optional (default = `0.0`)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            decoder_net: DecoderNet,
            max_decoding_steps: int,
            target_embedder: Embedding,
            target_namespace: str = "tokens",
            tie_output_embedding: bool = False,
            scheduled_sampling_ratio: float = 0,
            copy_token: str = "@COPY@",
            label_smoothing_ratio: Optional[float] = None,
            beam_size: int = 4,
            tensor_based_metric: Metric = None,
            token_based_metric: Metric = None,
    ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._decoder_net = decoder_net
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._oov_index = self._vocab.get_token_index(self._vocab._oov_token, self._target_namespace)
        self._copy_index = self._vocab.add_token_to_namespace(copy_token, self._target_namespace)
        self.target_embedder.extend_vocab(self._vocab, self._target_namespace)

        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        self._target_vocab_size = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        self.encoder_output_dim = self._decoder_net.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim
        self.decoder_input_dim = self._decoder_net.target_embedding_dim + self.decoder_output_dim

        self._input_projection_layer = Linear(
            self._decoder_net.target_embedding_dim + self.encoder_output_dim * 2,
            self.decoder_input_dim
        )

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(self.decoder_output_dim, self._target_vocab_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        # self._output_projection_layer = Linear(
        #     self._decoder_net.get_output_dim(), target_vocab_size
        # )

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        # These metrics will be updated during training and validation
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric

        self._scheduled_sampling_ratio = scheduled_sampling_ratio

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def _gather_extended_gold_tokens(
            self,
            target_tokens: torch.Tensor,
            source_token_ids: torch.Tensor,
            target_token_ids: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Modify the gold target tokens relative to the extended vocabulary.

        For gold targets that are OOV but were copied from the source, the OOV index
        will be changed to the index of the first occurence in the source sentence,
        offset by the size of the target vocabulary.

        # Parameters

        target_tokens : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length)`.
        source_token_ids : `torch.Tensor`
            Shape: `(batch_size, source_sequence_length)`.
        target_token_ids : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length)`.

        # Returns

        torch.Tensor
            Modified `target_tokens` with OOV indices replaced by offset index
            of first match in source sentence.
        """
        batch_size, target_sequence_length = target_tokens.size()
        source_sequence_length = source_token_ids.size(1)
        # Only change indices for tokens that were OOV in target vocab but copied from source.
        # shape: (batch_size, target_sequence_length)
        oov = target_tokens == self._oov_index
        # shape: (batch_size, target_sequence_length, source_sequence_length)
        expanded_source_token_ids = source_token_ids.unsqueeze(1).expand(
            batch_size, target_sequence_length, source_sequence_length
        )
        # shape: (batch_size, target_sequence_length, source_sequence_length)
        expanded_target_token_ids = target_token_ids.unsqueeze(-1).expand(
            batch_size, target_sequence_length, source_sequence_length
        )
        # shape: (batch_size, target_sequence_length, source_sequence_length)
        matches = expanded_source_token_ids == expanded_target_token_ids
        # shape: (batch_size, target_sequence_length)
        copied = matches.sum(-1) > 0
        # shape: (batch_size, target_sequence_length)
        mask = oov & copied
        # shape: (batch_size, target_sequence_length)
        first_match = ((matches.cumsum(-1) == 1) & matches).to(torch.uint8).argmax(-1)
        # shape: (batch_size, target_sequence_length)
        new_target_tokens = (
                target_tokens * ~mask + (first_match.long() + self._target_vocab_size) * mask
        )
        return new_target_tokens

    def forward(
            self,
            encoder_out: Dict[str, torch.LongTensor],
            target_tokens: TextFieldTensors = None,
            metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        source_token_ids = encoder_out["source_token_ids"]
        target_token_ids = encoder_out["target_token_ids"]
        # source_to_target = encoder_out["source_to_target"]

        state = encoder_out
        decoder_init_state = self._decoder_net.init_decoder_state(state)
        state.update(decoder_init_state)

        if target_tokens:
            # state = self._init_decoder_state(state)
            state_forward_loss = (
                state if self.training else {k: v.clone() for k, v in state.items() if v is not None}
            )
            output_dict = self._forward_loss(target_tokens, target_token_ids, state_forward_loss)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

        if not self.training:
            # state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]
                    # shape: (batch_size, target_sequence_length)
                    gold_tokens = self._gather_extended_gold_tokens(
                        target_tokens["tokens"]["tokens"], source_token_ids, target_token_ids
                    )
                    self._tensor_based_metric(best_predictions, gold_tokens)  # type: ignore
                if self._token_based_metric is not None:
                    predicted_tokens = self._get_predicted_tokens(
                        output_dict["predictions"], metadata, n_best=1
                    )
                    self._token_based_metric(  # type: ignore
                        predicted_tokens, metadata
                    )

        return output_dict

    def _forward_loss(
            self,
            target_tokens: TextFieldTensors,
            target_token_ids: torch.Tensor,
            state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size, target_sequence_length = target_tokens["tokens"]["tokens"].size()

        # shape: (batch_size, source_sequence_length)
        source_mask = state["source_mask"]

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1
        # We use this to fill in the copy index when the previous input was copied.
        # shape: (batch_size,)
        copy_input_choices = source_mask.new_full(
            (batch_size,), fill_value=self._copy_index, dtype=torch.long
        )
        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        # shape: (batch_size, source_sequence_length)
        selective_weights = state["decoder_hidden"].new_zeros(source_mask.size())

        # Indicates which tokens in the source sentence match the current target token.
        # shape: (batch_size, source_sequence_length)
        target_to_source = state["source_token_ids"].new_zeros(source_mask.size())

        # This is just a tensor of ones which we use repeatedly in `self._get_ll_contrib`,
        # so we create it once here to avoid doing it over-and-over.
        generation_scores_mask = state["decoder_hidden"].new_full(
            (batch_size, self._target_vocab_size), fill_value=1.0, dtype=torch.bool
        )

        step_log_likelihoods = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            input_choices = target_tokens["tokens"]["tokens"][:, timestep]
            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know
            # it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                # shape: (batch_size,)
                copied = (
                        (input_choices == self._oov_index) & (target_to_source.sum(-1) > 0)
                ).long()
                # shape: (batch_size,)
                input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                # shape: (batch_size, source_sequence_length)
                target_to_source = state["source_token_ids"] == target_token_ids[
                                                                :, timestep + 1
                                                                ].unsqueeze(-1)
            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)
            # Get generation scores for each token in the target vocab.
            # shape: (batch_size, target_vocab_size)
            generation_scores = self._get_generation_scores(state)
            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            # shape: (batch_size, source_sequence_length)
            copy_scores = self._get_copy_scores(state)
            # shape: (batch_size,)
            step_target_tokens = target_tokens["tokens"]["tokens"][:, timestep + 1]
            step_log_likelihood, selective_weights = self._get_ll_contrib(
                generation_scores,
                generation_scores_mask,
                copy_scores,
                step_target_tokens,
                target_to_source,
                source_mask,
            )
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

        # Gather step log-likelihoods.
        # shape: (batch_size, num_decoding_steps = target_sequence_length - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)
        # Get target mask to exclude likelihood contributions from timesteps after
        # the END token.
        # shape: (batch_size, target_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)
        # The first timestep is just the START token, which is not included in the likelihoods.
        # shape: (batch_size, num_decoding_steps)
        target_mask = target_mask[:, 1:]
        # Sum of step log-likelihoods.
        # shape: (batch_size,)
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        # The loss is the negative log-likelihood, averaged over the batch.
        loss = -log_likelihood.sum() / batch_size

        return {"loss": loss}

    def _decoder_step(
            self,
            last_predictions: torch.Tensor,
            selective_weights: torch.Tensor,
            state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # shape: (group_size, source_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"]
        # shape: (group_size, target_embedding_dim)
        embedded_input = self.target_embedder(last_predictions)
        # shape: (group_size, source_sequence_length)
        attentive_weights = self._decoder_net._attention(
            state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask
        )
        # shape: (group_size, encoder_output_dim)
        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
        # shape: (group_size, encoder_output_dim)
        selective_read = util.weighted_sum(state["encoder_outputs"], selective_weights)
        # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        state["decoder_hidden"], state["decoder_context"] = self._decoder_net._decoder_cell(
            projected_decoder_input.float(),
            (state["decoder_hidden"].float(), state["decoder_context"].float()),
        )

        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._output_generation_layer(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, source_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]
        # shape: (batch_size, source_sequence_length, decoder_output_dim)
        copy_projection = self._output_copying_layer(encoder_outputs)
        # shape: (batch_size, source_sequence_length, decoder_output_dim)
        copy_projection = torch.tanh(copy_projection)
        # shape: (batch_size, source_sequence_length)
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)
        return copy_scores

    def _get_ll_contrib(
            self,
            generation_scores: torch.Tensor,
            generation_scores_mask: torch.BoolTensor,
            copy_scores: torch.Tensor,
            target_tokens: torch.Tensor,
            target_to_source: torch.Tensor,
            source_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log-likelihood contribution from a single timestep.

        # Parameters

        generation_scores : `torch.Tensor`
            Shape: `(batch_size, target_vocab_size)`
        generation_scores_mask : `torch.BoolTensor`
            Shape: `(batch_size, target_vocab_size)`. This is just a tensor of 1's.
        copy_scores : `torch.Tensor`
            Shape: `(batch_size, source_sequence_length)`
        target_tokens : `torch.Tensor`
            Shape: `(batch_size,)`
        target_to_source : `torch.Tensor`
            Shape: `(batch_size, source_sequence_length)`
        source_mask : `torch.BoolTensor`
            Shape: `(batch_size, source_sequence_length)`

        # Returns

        Tuple[torch.Tensor, torch.Tensor]
            Shape: `(batch_size,), (batch_size, source_sequence_length)`
        """
        _, target_size = generation_scores.size()

        # The point of this mask is to just mask out all source token scores
        # that just represent padding. We apply the mask to the concatenation
        # of the generation scores and the copy scores to normalize the scores
        # correctly during the softmax.
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        mask = torch.cat((generation_scores_mask, source_mask), dim=-1)
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # Calculate the log probability (`copy_log_probs`) for each token in the source sentence
        # that matches the current target token. We use the sum of these copy probabilities
        # for matching tokens in the source sentence to get the total probability
        # for the target token. We also need to normalize the individual copy probabilities
        # to create `selective_weights`, which are used in the next timestep to create
        # a selective read state.
        # shape: (batch_size, source_sequence_length)
        copy_log_probs = (
                log_probs[:, target_size:]
                + (
                        target_to_source.to(log_probs.dtype) + util.tiny_value_of_dtype(log_probs.dtype)
                ).log()
        )
        # Since `log_probs[:, target_size]` gives us the raw copy log probabilities,
        # we use a non-log softmax to get the normalized non-log copy probabilities.
        selective_weights = util.masked_softmax(log_probs[:, target_size:], target_to_source)
        # This mask ensures that item in the batch has a non-zero generation probabilities
        # for this timestep only when the gold target token is not OOV or there are no
        # matching tokens in the source sentence.
        # shape: (batch_size, 1)
        gen_mask = (target_tokens != self._oov_index) | (target_to_source.sum(-1) == 0)
        log_gen_mask = (gen_mask + util.tiny_value_of_dtype(log_probs.dtype)).log().unsqueeze(-1)
        # Now we get the generation score for the gold target token.
        # shape: (batch_size, 1)
        generation_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)) + log_gen_mask
        # ... and add the copy score to get the step log likelihood.
        # shape: (batch_size, 1 + source_sequence_length)
        combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
        # shape: (batch_size,)
        step_log_likelihood = util.logsumexp(combined_gen_and_copy)

        return step_log_likelihood, selective_weights

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, source_sequence_length = state["source_mask"].size()
        # Initialize the copy scores to zero.
        state["copy_log_probs"] = (
                state["decoder_hidden"].new_zeros((batch_size, source_sequence_length))
                + util.tiny_value_of_dtype(state["decoder_hidden"].dtype)
        ).log()
        # shape: (batch_size,)
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )
        return {"predicted_log_probs": log_probabilities, "predictions": all_top_k_predictions}

    def take_step(
            self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.

        Since we are predicting tokens out of the extended vocab (target vocab + all unique
        tokens from the source sentence), this is a little more complicated that just
        making a forward pass through the model. The output log probs will have
        shape `(group_size, target_vocab_size + source_sequence_length)` so that each
        token in the target vocab and source sentence are assigned a probability.

        Note that copy scores are assigned to each source token based on their position, not unique value.
        So if a token appears more than once in the source sentence, it will have more than one score.
        Further, if a source token is also part of the target vocab, its final score
        will be the sum of the generation and copy scores. Therefore, in order to
        get the score for all tokens in the extended vocab at this step,
        we have to combine copy scores for re-occuring source tokens and potentially
        add them to the generation scores for the matching token in the target vocab, if
        there is one.

        So we can break down the final log probs output as the concatenation of two
        matrices, A: `(group_size, target_vocab_size)`, and B: `(group_size, source_sequence_length)`.
        Matrix A contains the sum of the generation score and copy scores (possibly 0)
        for each target token. Matrix B contains left-over copy scores for source tokens
        that do NOT appear in the target vocab, with zeros everywhere else. But since
        a source token may appear more than once in the source sentence, we also have to
        sum the scores for each appearance of each unique source token. So matrix B
        actually only has non-zero values at the first occurence of each source token
        that is not in the target vocab.

        # Parameters

        last_predictions : `torch.Tensor`
            Shape: `(group_size,)`

        state : `Dict[str, torch.Tensor]`
            Contains all state tensors necessary to produce generation and copy scores
            for next step.

        step : `int`
            The time step in beam search decoding.

        Notes
        -----
        `group_size` != `batch_size`. In fact, `group_size` = `batch_size * beam_size`.
        """
        _, source_sequence_length = state["source_to_target"].size()

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied,
        # there will be all zeros.
        # shape: (group_size,), (group_size, source_sequence_length)
        input_choices, selective_weights = self._get_input_and_selective_weights(
            last_predictions, state
        )
        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)
        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)
        # Get the un-normalized copy scores for each token in the source sentence,
        # excluding the start and end tokens.
        # shape: (group_size, source_sequence_length)
        copy_scores = self._get_copy_scores(state)
        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # shape: (group_size, source_sequence_length)
        source_mask = state["source_mask"]
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        mask = torch.cat(
            (
                generation_scores.new_full(generation_scores.size(), True, dtype=torch.bool),
                source_mask,
            ),
            dim=-1,
        )
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # shape: (group_size, target_vocab_size), (group_size, source_sequence_length)
        generation_log_probs, copy_log_probs = log_probs.split(
            [self._target_vocab_size, source_sequence_length], dim=-1
        )
        # Update copy_probs needed for getting the `selective_weights` at the next timestep.
        state["copy_log_probs"] = copy_log_probs
        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: (group_size, target_vocab_size + source_sequence_length)
        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, state)

        return final_log_probs, state

    def _get_input_and_selective_weights(
            self, last_predictions: torch.LongTensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Get input choices for the decoder and the selective copy weights.

        The decoder input choices are simply the `last_predictions`, except for
        target OOV predictions that were copied from source tokens, in which case
        the prediction will be changed to the COPY symbol in the target namespace.

        The selective weights are just the probabilities assigned to source
        tokens that were copied, normalized to sum to 1. If no source tokens were copied,
        there will be all zeros.

        # Parameters

        last_predictions : `torch.LongTensor`
            Shape: `(group_size,)`
        state : `Dict[str, torch.Tensor]`

        # Returns

        Tuple[torch.LongTensor, torch.Tensor]
            `input_choices` (shape `(group_size,)`) and `selective_weights`
            (shape `(group_size, source_sequence_length)`).
        """
        group_size, source_sequence_length = state["source_to_target"].size()

        # This is a mask indicating which last predictions were copied from the
        # the source AND not in the target vocabulary (OOV).
        # (group_size,)
        only_copied_mask = last_predictions >= self._target_vocab_size

        # If the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        copy_input_choices = only_copied_mask.new_full(
            (group_size,), fill_value=self._copy_index, dtype=torch.long
        )
        input_choices = last_predictions * ~only_copied_mask + copy_input_choices * only_copied_mask

        # In order to get the `selective_weights`, we need to find out which predictions
        # were copied or copied AND generated, which is the case when a prediction appears
        # in both the source sentence and the target vocab. But whenever a prediction
        # is in the target vocab (even if it also appeared in the source sentence),
        # its index will be the corresponding target vocab index, not its index in
        # the source sentence offset by the target vocab size. So we first
        # use `state["source_to_target"]` to get an indicator of every source token
        # that matches the predicted target token.
        # shape: (group_size, source_sequence_length)
        expanded_last_predictions = last_predictions.unsqueeze(-1).expand(
            group_size, source_sequence_length
        )
        # shape: (group_size, source_sequence_length)
        source_copied_and_generated = state["source_to_target"] == expanded_last_predictions

        # In order to get indicators for copied source tokens that are OOV with respect
        # to the target vocab, we'll make use of `state["source_token_ids"]`.
        # First we adjust predictions relative to the start of the source tokens.
        # This makes sense because predictions for copied tokens are given by the index of the copied
        # token in the source sentence, offset by the size of the target vocabulary.
        # shape: (group_size,)
        adjusted_predictions = last_predictions - self._target_vocab_size
        # The adjusted indices for items that were not copied will be negative numbers,
        # and therefore invalid. So we zero them out.
        adjusted_predictions = adjusted_predictions * only_copied_mask
        # shape: (group_size, source_sequence_length)
        source_token_ids = state["source_token_ids"]
        # shape: (group_size, source_sequence_length)
        adjusted_prediction_ids = source_token_ids.gather(-1, adjusted_predictions.unsqueeze(-1))
        # This mask will contain indicators for source tokens that were copied
        # during the last timestep.
        # shape: (group_size, source_sequence_length)
        source_only_copied = source_token_ids == adjusted_prediction_ids
        # Since we zero'd-out indices for predictions that were not copied,
        # we need to zero out all entries of this mask corresponding to those predictions.
        source_only_copied = source_only_copied & only_copied_mask.unsqueeze(-1)

        # shape: (group_size, source_sequence_length)
        mask = source_only_copied | source_copied_and_generated
        # shape: (group_size, source_sequence_length)
        selective_weights = util.masked_softmax(state["copy_log_probs"], mask)

        return input_choices, selective_weights

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(
                    self._tensor_based_metric.get_metric(reset=reset)  # type: ignore
                )
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    def _gather_final_log_probs(
            self,
            generation_log_probs: torch.Tensor,
            copy_log_probs: torch.Tensor,
            state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine copy probabilities with generation probabilities for matching tokens.

        # Parameters

        generation_log_probs : `torch.Tensor`
            Shape: `(group_size, target_vocab_size)`
        copy_log_probs : `torch.Tensor`
            Shape: `(group_size, source_sequence_length)`
        state : `Dict[str, torch.Tensor]`

        # Returns

        torch.Tensor
            Shape: `(group_size, target_vocab_size + source_sequence_length)`.
        """
        _, source_sequence_length = state["source_to_target"].size()
        source_token_ids = state["source_token_ids"]

        # shape: [(batch_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(source_sequence_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            # `source_to_target` is a matrix of shape (group_size, source_sequence_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            source_to_target_slice = state["source_to_target"][:, i]
            # The OOV index in the source_to_target_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score
            # to the OOV token.
            copy_log_probs_to_add_mask = source_to_target_slice != self._oov_index
            copy_log_probs_to_add = (
                    copy_log_probs_slice
                    + (
                            copy_log_probs_to_add_mask
                            + util.tiny_value_of_dtype(copy_log_probs_slice.dtype)
                    ).log()
            )
            # shape: (batch_size, 1)
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)

            # shape: (batch_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(
                1, source_to_target_slice.unsqueeze(-1)
            )

            combined_scores = util.logsumexp(
                torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1)
            )

            generation_log_probs = generation_log_probs.scatter(
                -1, source_to_target_slice.unsqueeze(-1), combined_scores.unsqueeze(-1)
            )
            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurence of this particular source token, we add the log_probs from all other
            # occurences, otherwise we zero it out since it was already accounted for.
            if i < (source_sequence_length - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (group_size, source_sequence_length - i)
                source_future_occurences = source_token_ids[:, (i + 1):] == source_token_ids[
                                                                            :, i
                                                                            ].unsqueeze(-1)
                # shape: (group_size, source_sequence_length - i)
                future_copy_log_probs = (
                        copy_log_probs[:, (i + 1):]
                        + (
                                source_future_occurences + util.tiny_value_of_dtype(copy_log_probs.dtype)
                        ).log()
                )
                # shape: (group_size, 1 + source_sequence_length - i)
                combined = torch.cat(
                    (copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs), dim=-1
                )
                # shape: (group_size,)
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurences = source_token_ids[:, 0:i] == source_token_ids[
                                                                         :, i
                                                                         ].unsqueeze(-1)
                # shape: (group_size,)
                duplicate_mask = source_previous_occurences.sum(dim=-1) == 0
                copy_log_probs_slice = (
                        copy_log_probs_slice
                        + (duplicate_mask + util.tiny_value_of_dtype(copy_log_probs_slice.dtype)).log()
                )

            # Finally, we zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_log_probs = (
                    copy_log_probs_slice
                    + (
                            ~copy_log_probs_to_add_mask
                            + util.tiny_value_of_dtype(copy_log_probs_slice.dtype)
                    ).log()
            )
            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (group_size, target_vocab_size + source_sequence_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def _get_predicted_tokens(
            self,
            predicted_indices: Union[torch.Tensor, numpy.ndarray],
            batch_metadata: List[Any],
            n_best: int = None,
    ) -> List[Union[List[List[str]], List[str]]]:
        """
        Convert predicted indices into tokens.

        If `n_best = 1`, the result type will be `List[List[str]]`. Otherwise the result
        type will be `List[List[List[str]]]`.
        """
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens: List[Union[List[List[str]], List[str]]] = []
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens: List[List[str]] = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                indices = list(indices)
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                for index in indices:
                    if index >= self._target_vocab_size:
                        adjusted_index = index - self._target_vocab_size
                        token = metadata["source_tokens"][adjusted_index]
                    else:
                        token = self._vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        predicted_tokens = self._get_predicted_tokens(
            output_dict["predictions"], output_dict["metadata"]
        )
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict
