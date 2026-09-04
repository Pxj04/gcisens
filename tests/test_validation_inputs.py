import warnings

import numpy as np
import pandas as pd
import pytest
from pymcdm.methods import COMET, SPOTIS

from gcisens import (
    ESPExpert,
    SobolStudy,
    esp_comet,
    esp_spotis,
    sobol_analysis,
    validate_scores,
)
from gcisens.adapters import make_adapter


@pytest.mark.parametrize("argument", ["weights", "types"])
def test_comet_rejects_spotis_arguments(argument):
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    expert = ESPExpert(esps=np.array([[0.5, 0.5]]), bounds=bounds)
    model = COMET(expert.make_cvalues_psi(), expert)

    with pytest.raises(
        ValueError,
        match="COMET models do not take weights/types; they are estimated by regression",
    ):
        SobolStudy(model, bounds=bounds, **{argument: [0.5, 0.5]})


@pytest.mark.parametrize("weights", [[0.8, 0.3], [1.1, -0.1]])
def test_esp_spotis_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights must be non-negative and sum to 1"):
        esp_spotis(
            esp=[0.5, 0.5],
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            weights=weights,
        )


def test_spotis_adapter_rejects_invalid_weights():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    model = SPOTIS(bounds, esp=np.array([0.5, 0.5]))

    with pytest.raises(ValueError, match="weights must be non-negative and sum to 1"):
        SobolStudy(model, weights=[5.0, 3.0], n_samples=8)


@pytest.mark.parametrize("weights", [[0.8, 0.8], [1.2, -0.2]])
def test_callable_adapter_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights must be non-negative and sum to 1"):
        SobolStudy(
            lambda X: np.asarray(X)[:, 0],
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            weights=weights,
            n_samples=8,
        )


def test_esp_comet_rejects_esps_with_wrong_number_of_criteria():
    with pytest.raises(ValueError, match=r"esps must have shape \(k, 2\)"):
        esp_comet(esps=[[0.5, 0.5, 0.5]], bounds=[[0.0, 1.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    ("builder", "point_argument"),
    [(esp_comet, {"esps": [[0.5, 0.5]]}), (esp_spotis, {"esp": [0.5, 0.5]})],
)
def test_builders_reject_wrong_number_of_criteria_names(builder, point_argument):
    with pytest.raises(ValueError, match="Got 1 criteria names for 2 criteria"):
        builder(
            **point_argument,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            criteria_names=["only one"],
        )


def _linear_result():
    def score(X):
        X = np.atleast_2d(X)
        return 0.7 * X[:, 0] + 0.3 * X[:, 1]

    return SobolStudy(
        score,
        bounds=[[0.0, 10.0], [0.0, 10.0]],
        criteria_names=["A", "B"],
        weights=[0.7, 0.3],
        n_samples=8,
        second_order=False,
        seed=0,
    ).run()


def test_validate_reorders_named_dataframe_columns():
    result = _linear_result()
    X = pd.DataFrame({"A": [1.0, 8.0, 2.0, 9.0], "B": [9.0, 2.0, 8.0, 1.0]})
    labels = [False, True, False, True]

    original = result.validate(X, labels, top_k=[2])
    reordered = result.validate(X[["B", "A"]], labels, top_k=[2])

    np.testing.assert_allclose(reordered.scores, original.scores)
    pd.testing.assert_frame_equal(reordered.lift, original.lift)


def test_validate_rejects_wrong_number_of_columns():
    result = _linear_result()

    with pytest.raises(ValueError, match="X must have 2 columns, got 1"):
        result.validate([[1.0], [2.0]], labels=[False, True], top_k=[1])


def test_validate_reports_comet_criterion_outside_bounds():
    model = esp_comet(
        esps=[[0.5, 0.5]],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        criteria_names=["A", "B"],
    )
    result = SobolStudy(model, n_samples=8, second_order=False, seed=0).run()

    with pytest.raises(
        ValueError,
        match=r"Criterion 'B' contains value 1.5 outside bounds \[0.0, 1.0\]",
    ):
        result.validate([[0.2, 0.3], [0.4, 1.5]], labels=[False, True], top_k=[1])


def test_comet_error_reports_model_domain_when_sampling_bounds_differ():
    model_bounds = np.array([[0.0, 1.0]])
    expert = ESPExpert(esps=np.array([[0.5]]), bounds=model_bounds)
    model = COMET(expert.make_cvalues_psi(), expert)
    adapter = make_adapter(model, bounds=[[0.0, 2.0]], criteria_names=["A"])

    with pytest.raises(
        ValueError,
        match=r"Criterion 'A' contains value 1.5 outside bounds \[0.0, 1.0\]",
    ):
        adapter.scores([[1.5]])


@pytest.mark.parametrize(
    ("labels", "missing_group"),
    [([False, False], "positive"), ([True, True], "negative")],
)
def test_validate_scores_rejects_missing_label_group(labels, missing_group):
    with pytest.raises(
        ValueError,
        match=f"labels must contain at least one {missing_group}",
    ):
        validate_scores([0.1, 0.2], labels, top_k=[1])


def test_validate_scores_caps_top_k_at_sample_count():
    result = validate_scores(
        scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        labels=[False, True, False, True, False],
        top_k=[500],
    )

    assert result.lift.loc[0, "k"] == 5
    assert result.lift.loc[0, "rate"] == pytest.approx(0.4)
    assert result.lift.loc[0, "lift"] == pytest.approx(1.0)


def test_validate_scores_single_member_std_is_nan_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = validate_scores(
            scores=[0.1, 0.2, 0.3],
            labels=[True, False, False],
            top_k=[1],
        )

    assert np.isnan(result.groups.loc[0, "std_score"])


@pytest.mark.parametrize(
    "call",
    [
        lambda: esp_comet(esps=[[1.0]], bounds=[[1.0, 1.0]]),
        lambda: esp_spotis(esp=[1.0], bounds=[[1.0, 1.0]]),
        lambda: SobolStudy(lambda X: X[:, 0], bounds=[[1.0, 1.0]], n_samples=8),
        lambda: sobol_analysis(lambda X: X[:, 0], [[1.0, 1.0]], ["A"], n_samples=8),
    ],
)
def test_public_entry_points_reject_invalid_bounds_rows(call):
    with pytest.raises(
        ValueError,
        match=r"bounds row 0 must have min < max, got \[1.0, 1.0\]",
    ):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: SobolStudy(lambda X: X[:, 0], bounds=[[0.0, 1.0]], n_samples=1),
        lambda: sobol_analysis(lambda X: X[:, 0], [[0.0, 1.0]], ["A"], n_samples=1),
    ],
)
def test_sampling_entry_points_reject_too_few_samples(call):
    with pytest.raises(ValueError, match="n_samples must be at least 2, got 1"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: SobolStudy(lambda X: X[:, 0], bounds=[[0.0, 1.0]], n_samples=30),
        lambda: sobol_analysis(
            lambda X: X[:, 0],
            [[0.0, 1.0]],
            ["A"],
            n_samples=30,
            second_order=False,
        ),
    ],
)
def test_sampling_entry_points_warn_for_non_power_of_two(call):
    with pytest.warns(
        UserWarning,
        match="n_samples=30 is not a power of two; Sobol' convergence may be reduced",
    ):
        call()


def test_sobol_study_warns_once_for_non_power_of_two():
    with pytest.warns(UserWarning, match="n_samples=30") as caught:
        SobolStudy(
            lambda X: X[:, 0],
            bounds=[[0.0, 1.0]],
            n_samples=30,
            second_order=False,
        ).run()

    # Count only the gcisens warning: old SALib/pandas combinations add
    # unrelated FutureWarnings inside the same block.
    ours = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "n_samples=30" in str(w.message)
    ]
    assert len(ours) == 1


def test_sobol_study_rejects_unknown_sampler_at_construction():
    with pytest.raises(ValueError, match="sampler must be one of"):
        SobolStudy(
            lambda X: X[:, 0],
            bounds=[[0.0, 1.0]],
            n_samples=8,
            sampler="lhs",
        )


@pytest.mark.parametrize("types", [[1], [0, 0], [1, np.nan], [[1, -1]]])
def test_spotis_types_checked_in_builder_and_plain_model(types):
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="types"):
        esp_spotis([0.5, 0.5], bounds, types=types)
    with pytest.raises(ValueError, match="types"):
        make_adapter(SPOTIS(bounds), weights=[0.5, 0.5], types=types)


@pytest.mark.parametrize("names", [["A", "A"], [1, "1"], ["", "B"], "AB"])
def test_duplicate_or_empty_names_fail_before_building(names):
    with pytest.raises(ValueError, match="criteria_names"):
        esp_spotis([0.5, 0.5], [[0, 1], [0, 1]], criteria_names=names)
    with pytest.raises(ValueError, match="criteria_names"):
        sobol_analysis(lambda X: X[:, 0], [[0, 1], [0, 1]], names, n_samples=8)


@pytest.mark.parametrize("builder", [esp_comet, esp_spotis])
def test_builders_reject_nonfinite_esps(builder):
    with pytest.raises(ValueError, match="finite"):
        builder([np.nan, 0.5], [[0, 1], [0, 1]])


@pytest.mark.parametrize("top_k", [[0], [-1], [1.5], [True], [], 1])
def test_invalid_lift_cutoffs_are_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        validate_scores([0.1, 0.2], [False, True], top_k=top_k)


@pytest.mark.parametrize("labels", [[0, 2], [0, np.nan], ["No", "Yes"], [0, np.inf]])
def test_validation_requires_binary_labels(labels):
    with pytest.raises(ValueError, match="binary"):
        validate_scores([0.1, 0.2], labels, top_k=[1])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_validation_requires_finite_scores(value):
    with pytest.raises(ValueError, match="finite"):
        validate_scores([0.1, value], [0, 1], top_k=[1])


def test_validation_result_owns_readonly_scores_and_labels():
    from dataclasses import FrozenInstanceError

    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([False, False, True])
    result = validate_scores(scores, labels, top_k=[1])
    groups, lift = result.groups, result.lift
    scores[:] = 0
    labels[:] = False
    np.testing.assert_array_equal(result.scores, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(result.labels, [False, False, True])
    for values in (result.scores, result.labels):
        with pytest.raises(ValueError):
            values[0] = 0
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.delta_mean = 0
    pd.testing.assert_frame_equal(result.groups, groups)
    pd.testing.assert_frame_equal(result.lift, lift)


def test_validation_tables_are_detached_and_record_can_be_copied():
    from copy import copy

    from gcisens.validation import ValidationResult

    result = validate_scores([0.1, 0.2, 0.3], [0, 0, 1], top_k=[1])
    groups, lift = result.groups, result.lift
    copied = ValidationResult(groups, lift, result.delta_mean, result.scores, result.labels)
    groups.loc[0, "mean_score"] = -99
    lift.loc[0, "lift"] = -99
    returned_groups, returned_lift = copied.groups, copied.lift
    returned_groups.loc[0, "mean_score"] = -99
    returned_lift.loc[0, "lift"] = -99
    pd.testing.assert_frame_equal(copied.groups, result.groups)
    pd.testing.assert_frame_equal(copied.lift, result.lift)
    shallow_copy = copy(copied)
    pd.testing.assert_frame_equal(shallow_copy.groups, result.groups)
    pd.testing.assert_frame_equal(shallow_copy.lift, result.lift)
