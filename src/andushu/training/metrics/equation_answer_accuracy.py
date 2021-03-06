from typing import List, Dict

from overrides import overrides

from allennlp.training.metrics import Metric

from andushu.dataset_readers.math.equation2tree import eval_tree


@Metric.register("equation_answer_accuracy")
class EquationAnswerAccuracy(Metric):
    """
    Simple sequence accuracy based on tokens, as opposed to tensors.
    """

    def __init__(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 gold_targets: List[List[str]]) -> None:
        self._total_counts += len(predictions)
        for predicted_tokens, gold_tokens in zip(predictions, gold_targets):
            if predicted_tokens == gold_tokens:
                self._correct_counts += 1
            else:
                try:
                    pred = eval_tree(' '.join(predicted_tokens))
                    ans = eval_tree(' '.join(gold_tokens))
                    if abs(pred - ans) < 1e-9:
                        self._correct_counts += 1
                    # else:
                    #     print(predicted_tokens, gold_tokens)
                except Exception as e:
                    # print(e, predicted_tokens, gold_tokens)
                    pass

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"answer_acc": accuracy}
