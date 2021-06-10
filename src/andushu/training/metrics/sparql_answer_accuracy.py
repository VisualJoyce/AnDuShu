from typing import List, Dict, Any

from allennlp.training.metrics import Metric
from more_itertools import flatten
from overrides import overrides

from andushu.dataset_readers.kbqa.minisparql import IndexedTripleStore, TripleStore, parse_data


@Metric.register("sparql_answer_accuracy")
class SparqlAnswerAccuracy(Metric):
    """
    Simple sequence accuracy based on tokens, as opposed to tensors.
    """

    def __init__(self, triples_data_path, debug=False) -> None:
        self.triples_data_path = triples_data_path
        self.debug = debug
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

        use_index = False
        use_lower = True
        if use_index:
            store = IndexedTripleStore(use_lower)
        else:
            store = TripleStore(use_lower)
        store.import_file(open(self.triples_data_path))

        # for pred, meta in zip(predicted_tokens, metadata):
        for i, (predicted_tokens, meta) in enumerate(zip(predictions, metadata)):
            gold_tokens = meta['target_tokens']
            if predicted_tokens == gold_tokens:
                self._correct_counts += 1
            else:
                query_result = []
                for script in ' '.join(predicted_tokens).split(';'):
                    try:
                        q = store.query(script)
                        columns = [k.strip('?') for k in script.split() if k.startswith('?')]
                        query_result.extend(parse_data(q, columns).value.str.strip("'").tolist())
                    except Exception as e:
                        pass
                        # print(e, meta, script)

                ground_truth = set(meta['answer'].strip('|').split('|'))
                query_result = set(flatten([item.strip('|').split('|') for item in query_result]))
                if query_result != ground_truth:
                    if i == 0 and self.debug:
                        print('--------------------------')
                        print(meta)
                        print(predicted_tokens)
                        print(query_result, ground_truth)
                else:
                    self._correct_counts += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"answer_acc": accuracy}
