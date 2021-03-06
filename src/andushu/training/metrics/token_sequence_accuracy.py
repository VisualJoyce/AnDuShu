from typing import List, Dict, Any

from overrides import overrides

from allennlp.training.metrics import Metric


@Metric.register("token_sequence_accuracy")
class TokenSequenceAccuracy(Metric):
    """
    Simple sequence accuracy based on tokens, as opposed to tensors.
    """

    def __init__(self, ) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 metadata: List[Dict[str, Any]]) -> None:
        self._total_counts += len(predictions)
        for predicted_tokens, meta in zip(predictions, metadata):
            gold_tokens = meta['target_tokens']
            if predicted_tokens == gold_tokens:
                self._correct_counts += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"seq_acc": accuracy}
