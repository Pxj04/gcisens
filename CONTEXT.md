# gcisens terminology

Use this vocabulary in code, documentation and article text. See
`CONTRIBUTING.md` for the module map and release procedure.

## Workflow

Users pass a pymcdm COMET or SPOTIS model to `SobolStudy(...).run()`. They
inspect criterion weights and Sobol' indices with `result.table()` and save
outputs through result methods. Builders are optional shortcuts. Keep helper
records and adapters out of the main user workflow.

## Criterion importance

**Weight, w** is a declared input for SPOTIS and a regression estimate for
COMET. Always state the source. Do not call COMET weights declared weights.

**Local weight, w_loc** comes from a conditional range sweep. Vary one
criterion over its entire bounds and hold the other criteria at the reference
point. This does not measure only a small neighbourhood of that point.

**Reference point** fixes the other criteria during a local-weight sweep.

**Sobol' indices, S1, ST and S2** describe first-order, total-order and pairwise
interaction contributions to score variance. This implementation uses
independent uniform inputs over the criterion bounds.

**Rank** is a criterion's position by weight or index, with 1 most important.
Exact ties share their average rank. These are criterion ranks, not ranks of
alternatives.

**View** is a compatibility record for a criterion measure and its display
settings. It is not needed to run a study.

## Diagnosis

The **Sensitivity Discrepancy Report** assigns one category per criterion
from weights, S1 and ST. Rules use dimensionless thresholds and point estimates.
Bootstrap confidence intervals are reported separately.

The retained category **confirmed transparency** means no discrepancy was
found at the chosen thresholds. It is not proof of transparency.

**Rank displacement** is the distance between a criterion's weight rank and
S1 rank. It is one trigger of moderate discrepancy.

## Models

An **Expected Solution Point, ESP** is a point the decision maker considers
ideal. ESP-COMET and ESP-SPOTIS use it to score alternatives.

**Score orientation** states whether higher or lower scores mean closer to the
ESP. COMET uses higher preferences; SPOTIS uses lower distances.

A **custom scoring function** must return one finite score per row, give the
same point the same score regardless of other rows, and be deterministic.
Batch-dependent normalisation or refitting does not meet this contract.
