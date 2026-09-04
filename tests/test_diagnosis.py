import numpy as np

from gcisens.diagnosis import (
    CONFIRMED_TRANSPARENCY,
    HIDDEN_INFLUENCE,
    INTERACTION_DOMINANCE,
    MODERATE_DISCREPANCY,
    DiagnosisThresholds,
    classify,
)

NAMES = ["C1", "C2", "C3", "C4"]


def categories(w, s1, st, thresholds=None):
    return [
        d.category for d in classify(NAMES, np.array(w), np.array(s1), np.array(st), thresholds)
    ]


def test_confirmed_transparency():
    w = [0.4, 0.3, 0.2, 0.1]
    s1 = [0.41, 0.29, 0.21, 0.09]
    st = [0.42, 0.30, 0.21, 0.10]
    assert categories(w, s1, st) == [CONFIRMED_TRANSPARENCY] * 4


def test_hidden_influence_detected():
    # C4: near-zero weight, substantial total-order effect.
    w = [0.5, 0.3, 0.19, 0.01]
    s1 = [0.45, 0.28, 0.17, 0.06]
    st = [0.46, 0.29, 0.18, 0.07]
    assert categories(w, s1, st)[3] == HIDDEN_INFLUENCE


def test_interaction_dominance_detected():
    # C1: ST far above S1.
    w = [0.4, 0.3, 0.2, 0.1]
    s1 = [0.30, 0.29, 0.21, 0.09]
    st = [0.45, 0.30, 0.21, 0.10]
    assert categories(w, s1, st)[0] == INTERACTION_DOMINANCE


def test_hidden_influence_wins_over_interaction():
    # Near-zero weight acting through interaction: reported as hidden (rule order).
    w = [0.55, 0.44, 0.005, 0.005]
    s1 = [0.5, 0.4, 0.01, 0.01]
    st = [0.5, 0.4, 0.09, 0.01]
    assert categories(w, s1, st)[2] == HIDDEN_INFLUENCE


def test_moderate_discrepancy_by_rank_displacement():
    # C1 ranked 1st by weight but 3rd by S1 (displacement 2), no interactions.
    w = [0.35, 0.30, 0.25, 0.10]
    s1 = [0.20, 0.32, 0.38, 0.10]
    st = [0.21, 0.33, 0.38, 0.10]
    cats = categories(w, s1, st)
    assert cats[0] == MODERATE_DISCREPANCY
    assert cats[2] == MODERATE_DISCREPANCY


def test_moderate_discrepancy_by_dismissed_criterion():
    # C4: weight ~0 dismisses a still-influential criterion (ST just above floor,
    # but below the hidden-influence excess).
    w = [0.4, 0.35, 0.245, 0.005]
    s1 = [0.4, 0.33, 0.22, 0.025]
    st = [0.4, 0.34, 0.23, 0.028]
    assert categories(w, s1, st)[3] == MODERATE_DISCREPANCY


def test_equal_weights_do_not_trigger_arbitrary_displacement():
    # Exactly tied weights get an average rank, not input-order ranks; with
    # sensitivity ranks 1..4 the max displacement from 2.5 is 1.5 < 2.
    w = [0.25, 0.25, 0.25, 0.25]
    s1 = [0.30, 0.27, 0.23, 0.20]
    st = [0.30, 0.27, 0.23, 0.20]
    assert categories(w, s1, st) == [CONFIRMED_TRANSPARENCY] * 4


def test_thresholds_are_configurable():
    w = [0.4, 0.3, 0.2, 0.1]
    s1 = [0.30, 0.29, 0.21, 0.09]
    st = [0.45, 0.30, 0.21, 0.10]
    strict = DiagnosisThresholds(interaction_ratio=0.9, interaction_abs=0.5)
    assert categories(w, s1, st, strict)[0] != INTERACTION_DOMINANCE


def test_nonfinite_indices_cannot_receive_a_diagnosis():
    import pytest

    for value in (np.nan, np.inf, -np.inf):
        for field in ("S1", "ST"):
            inputs = {"S1": [0.8, 0.2], "ST": [0.8, 0.2], field: [0.8, value]}
            with pytest.raises(ValueError, match=field):
                classify(["A", "B"], [0.8, 0.2], **inputs)


def test_finite_negative_estimates_are_allowed():
    result = classify(["A", "B"], [1.0, 0.0], [1.01, -0.01], [1.0, -0.002])
    assert len(result) == 2


def test_transparency_detail_states_threshold_limit():
    diagnosis = classify(["A"], [1.0], [1.0], [1.0])[0]
    assert "no discrepancy detected under the chosen thresholds" in diagnosis.detail


def test_thresholds_are_validated_and_cannot_change():
    from dataclasses import FrozenInstanceError

    import pytest

    for settings in ({"hidden_st_excess": np.nan}, {"s2_min_abs": -1}, {"rank_displacement": 1.5}):
        with pytest.raises(ValueError):
            DiagnosisThresholds(**settings)
    thresholds = DiagnosisThresholds()
    with pytest.raises(FrozenInstanceError):
        thresholds.hidden_st_excess = 0.5
