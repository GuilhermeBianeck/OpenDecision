import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from opendecision import DecisionModel, DecisionRequest
from opendecision.confidence import softmax
from opendecision.errors import BackendError


class FixedBackend:
    name = "fixture"
    model_id = "fixture-v1"
    revision = "abc123"
    device = "cpu"
    precision = "float32"

    def __init__(self, scores=None):
        self.scores = scores or [3.0, 1.0]
        self.requests = []

    def score_batch(self, requests):
        self.requests.extend(requests)
        return [self.scores[:] for _ in requests]


class StatementBackend(FixedBackend):
    """Fixture returning fixed [entailment, neutral, contradiction] logits."""

    supports_statements = True

    def __init__(self, logits=None):
        super().__init__()
        self.logits = logits or [2.0, 0.0, -1.0]
        self.statements = []

    def score_statements(self, requests):
        self.statements.extend(requests)
        return [self.logits[:] for _ in requests]


def choose(backend=None, **kwargs):
    return DecisionModel(backend=backend or FixedBackend()).choose(
        state="A public test fixture", question="Which?", choices=["a", "b"], **kwargs
    )


@pytest.mark.parametrize(
    "choices", [["a"], ["a", "a"], ["a", " a "], [" ", "b"], ["x" * 4097, "b"]]
)
def test_invalid_choices(choices):
    with pytest.raises(ValidationError):
        DecisionRequest(state="", question="Which?", choices=choices)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValidationError):
        choose(abstain_threshold=threshold)


def test_extreme_softmax_is_finite_and_translation_invariant():
    assert softmax([10001.0, 10000.0]) == pytest.approx(softmax([1.0, 0.0]))
    assert sum(softmax([1e308, -1e308])) == 1.0
    assert all(math.isfinite(p) for p in softmax([-1e308, -1e308]))


@pytest.mark.parametrize("scores", [[1], [float("nan"), 2], [1, float("inf")]])
def test_invalid_backend_outputs_fail_closed(scores):
    with pytest.raises(BackendError):
        choose(FixedBackend(scores))


def test_normalized_not_calibrated_by_default():
    result = choose(include_raw_scores=True)
    assert result.choice == "a"
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert result.probabilities == result.normalized_probabilities
    assert result.calibrated_probabilities is None
    assert not result.metadata["calibrated"]
    assert result.raw_scores == {"a": 3, "b": 1}
    assert result.confidence == pytest.approx(result.probabilities["a"] - result.probabilities["b"])


def test_separate_abstention_thresholds_and_boundary():
    reference = choose()
    assert choose(abstain_threshold=reference.top_probability).choice == "a"
    assert choose(margin_threshold=reference.confidence).choice == "a"
    assert choose(abstain_threshold=0.99).abstained
    assert choose(margin_threshold=0.99).choice is None
    assert choose(abstain_threshold=0, margin_threshold=0).choice == "a"


def test_ties_are_stable_under_permutation_and_can_abstain():
    model = DecisionModel(backend=FixedBackend([0, 0]))
    assert model.choose(state="", question="?", choices=["b", "a"]).choice == "a"
    assert model.choose(state="", question="?", choices=["a", "b"]).choice == "a"
    assert model.choose(state="", question="?", choices=["a", "b"], margin_threshold=0.01).abstained


def test_independent_multilabel_is_not_one_softmax():
    backend = FixedBackend()
    result = DecisionModel(backend=backend).multi_label(state="", labels=["urgent?", "billing?"])
    assert len(backend.requests) == 2
    assert all(r.choices == ["yes", "no"] for r in backend.requests)
    assert all(label.value for label in result.values())
    assert sum(label.probability for label in result.values()) > 1


def test_boolean_abstention_and_ranking():
    model = DecisionModel(backend=FixedBackend())
    boolean = model.boolean(state="", question="?", abstain_threshold=0.999)
    assert boolean.value is None
    assert boolean.probability > 0.5
    assert boolean.method == "binary_choice"
    assert boolean.unsupported is None
    ranking = model.rank(state="", question="?", choices=["b", "a"], include_raw_scores=True)
    assert [x.choice for x in ranking.ranking] == ["b", "a"]
    assert ranking.ranking[0].raw_score == 3


def test_statement_backend_decides_between_support_and_contradiction():
    backend = StatementBackend([2.0, 0.0, -1.0])
    model = DecisionModel(backend=backend)
    assert model.supports_statements
    result = model.boolean(state="s", question="The invoice is paid.")
    assert result.method == "statement"
    assert result.value is True
    # Yes/no ignores the neutral logit; unsupported reports it separately.
    assert result.probability == pytest.approx(softmax([2.0, -1.0])[0])
    assert result.unsupported == pytest.approx(softmax([2.0, 0.0, -1.0])[1])
    assert result.decision.probabilities == {
        "yes": pytest.approx(result.probability),
        "no": pytest.approx(1 - result.probability),
    }
    assert result.decision.metadata["method"] == "statement"
    assert backend.statements[0].statement == "The invoice is paid."
    assert not backend.requests


def test_unsupported_threshold_abstains_only_on_neutral_mass():
    # Support dominates contradiction, but the neutral label carries most mass.
    model = DecisionModel(backend=StatementBackend([1.0, 4.0, -1.0]))
    confident = model.boolean(state="s", question="claim", unsupported_threshold=0.99)
    assert confident.value is True
    uncertain = model.boolean(state="s", question="claim", unsupported_threshold=0.5)
    assert uncertain.value is None
    assert uncertain.decision.abstained
    assert uncertain.probability == confident.probability
    assert uncertain.unsupported > 0.5
    boundary = model.boolean(
        state="s", question="claim", unsupported_threshold=uncertain.unsupported
    )
    assert boundary.value is True


def test_multi_label_uses_statements_when_available_and_falls_back_otherwise():
    backend = StatementBackend()
    result = DecisionModel(backend=backend).multi_label(
        state="s", labels=["urgent", "billing"], unsupported_threshold=0.9
    )
    assert [r.statement for r in backend.statements] == ["urgent", "billing"]
    assert all(r.method == "statement" for r in result.values())
    fallback = DecisionModel(backend=FixedBackend())
    assert fallback.multi_label(state="s", labels=["urgent"])["urgent"].method == "binary_choice"
    with pytest.raises(ValueError, match="unsupported_threshold requires statement scoring"):
        fallback.boolean(state="s", question="claim", unsupported_threshold=0.5)


@pytest.mark.parametrize("logits", [[1.0, 2.0], [float("nan"), 0.0, 0.0]])
def test_invalid_statement_logits_fail_closed(logits):
    with pytest.raises(BackendError, match="three finite logits"):
        DecisionModel(backend=StatementBackend(logits)).boolean(state="s", question="c")


def test_many_questions_and_batch_size_errors():
    model = DecisionModel(backend=FixedBackend())
    result = model.decide_many(
        state="shared", questions={"retry?": ["yes", "no"], "review?": ["yes", "no"]}
    )
    assert list(result) == ["retry?", "review?"]
    assert model.choose_batch([]) == []
    with pytest.raises(ValueError):
        model.decide_many(state="", questions={})
    with pytest.raises(ValueError):
        model.multi_label(state="", labels=["same", "same"])


def test_wrong_batch_cardinality_fails():
    class Broken(FixedBackend):
        def score_batch(self, requests):
            return []

    with pytest.raises(BackendError, match="number"):
        choose(Broken())


def test_model_can_be_shared_between_threads():
    model = DecisionModel(backend=FixedBackend())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: model.choose(state="", question="?", choices=["a", "b"]), range(20))
        )
    assert all(result.choice == "a" for result in results)
