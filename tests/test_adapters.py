"""The adapter answers for itself: declared weights with their source, ESPs,
grid lines and local weights come from the adapter, never from the model."""

import numpy as np
import pytest

from gcisens.adapters import CallableAdapter, CometAdapter, SpotisAdapter

BOUNDS = np.array([[0.0, 10.0], [0.0, 5.0]])


class FakeComet:
    """A COMET-shaped model: characteristic values, an expert with ESPs and
    a preference function that is additive in both criteria."""

    def __init__(self, esps=None):
        self.cvalues = [np.array([0.0, 4.0, 10.0]), np.array([0.0, 2.0, 5.0])]
        self.expert_function = _FakeExpert(esps)

    def __call__(self, X):
        X = np.atleast_2d(X)
        return 0.8 * X[:, 0] / 10 + 0.2 * X[:, 1] / 5


class _FakeExpert:
    def __init__(self, esps):
        self.esps = esps


class FakeSpotis:
    """A SPOTIS-shaped model: bounds, an ESP and a weighted distance."""

    def __init__(self, esp=None):
        self.bounds = BOUNDS
        self.esp = esp

    def __call__(self, X, weights, types, validation=False):
        X = np.atleast_2d(X)
        span = self.bounds[:, 1] - self.bounds[:, 0]
        return np.abs(X - self.esp) / span @ weights


def test_comet_adapter_declares_regression_weights_with_fit():
    adapter = CometAdapter(FakeComet())

    declared = adapter.declared_weights()

    assert declared.source == "regression (characteristic objects)"
    np.testing.assert_allclose(declared.weights, [0.8, 0.2], atol=1e-10)
    assert declared.r2 == pytest.approx(1.0)


def test_spotis_adapter_declares_its_input_weights_without_fit():
    adapter = SpotisAdapter(FakeSpotis(esp=[7.0, 2.0]), weights=[0.6, 0.4])

    declared = adapter.declared_weights()

    assert declared.source == "declared"
    np.testing.assert_allclose(declared.weights, [0.6, 0.4])
    assert declared.r2 is None


def test_callable_adapter_declares_nothing_without_weights():
    adapter = CallableAdapter(lambda X: np.atleast_2d(X)[:, 0], bounds=BOUNDS)

    assert adapter.declared_weights() is None
    assert (
        CallableAdapter(adapter.model, bounds=BOUNDS, weights=[0.5, 0.5]).declared_weights().source
        == "declared"
    )


def test_adapters_expose_esps_and_grid_lines_for_plotting():
    comet = CometAdapter(FakeComet(esps=np.array([[7.0, 2.0]])))
    spotis = SpotisAdapter(FakeSpotis(esp=[7.0, 2.0]), weights=[0.5, 0.5])
    bare = CallableAdapter(lambda X: np.atleast_2d(X)[:, 0], bounds=BOUNDS)
    given = CallableAdapter(bare.model, bounds=BOUNDS, esps=[[1.0, 1.0], [9.0, 4.0]])

    np.testing.assert_array_equal(comet.esps, [[7.0, 2.0]])
    np.testing.assert_array_equal(spotis.esps, [[7.0, 2.0]])
    assert bare.esps is None
    np.testing.assert_array_equal(given.esps, [[1.0, 1.0], [9.0, 4.0]])

    assert [list(v) for v in comet.grid_lines()] == [[0.0, 4.0, 10.0], [0.0, 2.0, 5.0]]
    assert spotis.grid_lines() is None
    assert bare.grid_lines() is None


def test_comet_adapter_without_esp_expert_has_no_esps():
    class PlainComet(FakeComet):
        def __init__(self):
            super().__init__()
            self.expert_function = lambda co: None

    assert CometAdapter(PlainComet()).esps is None


def test_local_weights_use_one_range_sweep_for_every_model(hr_setup):
    from pymcdm.methods.comet_tools import get_local_weights

    from gcisens import esp_comet
    from gcisens.weights import sweep_local_weights

    criteria, bounds, esp1, _ = hr_setup
    adapter = CometAdapter(esp_comet(esps=esp1, bounds=bounds, criteria_names=criteria))
    point = np.array([30, 10, 3000, 4, 15, 10, 5], dtype=float)

    local = adapter.local_weights(point, percent_step=0.05)

    np.testing.assert_allclose(local, sweep_local_weights(adapter.scores, point, bounds, 0.05))
    np.testing.assert_allclose(local, get_local_weights(adapter.model, point, 0.05))
    assert "local_weights" not in vars(CometAdapter)


def test_make_adapter_hands_builder_metadata_to_the_adapter():
    from gcisens import SobolStudy, esp_comet, esp_spotis

    comet = SobolStudy(esp_comet(esps=[[7, 2]], bounds=BOUNDS, criteria_names=["A", "B"])).adapter
    spotis = SobolStudy(
        esp_spotis(esp=[7, 2], bounds=BOUNDS, weights=[0.6, 0.4], criteria_names=["A", "B"])
    ).adapter

    assert comet.criteria_names == spotis.criteria_names == ["A", "B"]
    np.testing.assert_array_equal(comet.esps, [[7.0, 2.0]])
    np.testing.assert_array_equal(spotis.esps, [[7.0, 2.0]])
    np.testing.assert_allclose(spotis.weights, [0.6, 0.4])


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("weights", [0.7, 0.3]),
        ("criteria_names", ["X", "Y"]),
        ("bounds", [[0.0, 10.0], [0.0, 6.0]]),
        ("types", [1, -1]),
    ],
)
def test_explicit_study_arguments_that_conflict_with_builder_metadata_raise(argument, value):
    from gcisens import SobolStudy, esp_spotis

    model = esp_spotis(esp=[7, 2], bounds=BOUNDS, weights=[0.6, 0.4], criteria_names=["A", "B"])

    with pytest.raises(ValueError, match=argument):
        SobolStudy(model, **{argument: value})


def test_explicit_study_arguments_equal_to_builder_metadata_are_accepted():
    from gcisens import SobolStudy, esp_spotis

    model = esp_spotis(esp=[7, 2], bounds=BOUNDS, weights=[0.6, 0.4], criteria_names=["A", "B"])

    adapter = SobolStudy(
        model, bounds=BOUNDS.tolist(), weights=[0.6, 0.4], criteria_names=["A", "B"]
    ).adapter

    np.testing.assert_allclose(adapter.weights, [0.6, 0.4])


def test_only_the_adapter_modules_know_the_model_internals():
    import inspect
    import re

    from gcisens import diagnosis, export, plots, sensitivity, study, validation, weights

    for module in (study, plots, export, weights, sensitivity, diagnosis, validation):
        source = inspect.getsource(module)
        assert "declared_weights_r2" not in source
        assert not re.search(r"isinstance\([^)]*COMET", source)
        assert "META_ATTR" not in source
        assert "_gcisens_meta" not in source
        assert "adapter.model" not in source
        assert ".cvalues" not in source
        assert "expert_function" not in source
