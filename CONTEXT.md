# gcisens

Variance-based sensitivity analysis of MCDA scoring models (ESP-COMET, ESP-SPOTIS), and the Sensitivity Discrepancy Report that compares what a model declares about its criteria with what the model actually does.

## Language

### Criteria importance

**View**:
One account of how important each criterion is: a value per criterion and the ranking those values induce. A study has an ordered set of views: w, w_loc (only with a reference point), S1 and ST.
_Avoid_: ranking (a view induces one), column, series, perspective

**Declared weight (w)**:
The importance a model reports to stakeholders: an input for SPOTIS, estimated by regression on characteristic objects for COMET.
_Avoid_: global weight, nominal weight

**Local weight (w_loc)**:
Criterion importance in the neighbourhood of a reference point, from a one-criterion range sweep.

**Reference point**:
A point in the criteria space at which local weights are computed.
_Avoid_: anchor, probe point

**Sobol' index (S1, ST, S2)**:
The first-order, total-order or pairwise-interaction share of score variance attributable to a criterion.

**Rank**:
A criterion's position within a view, 1 = most important. Exact ties share their average rank. The whole package has one rank definition.
_Avoid_: ordinal rank, position

### Diagnosis

**Sensitivity Discrepancy Report**:
A per-criterion category that compares the declared weight with the observed Sobol' indices: hidden influence, interaction dominance, moderate discrepancy or confirmed transparency.
_Avoid_: diagnosis table, classification report

**Rank displacement**:
The distance between a criterion's rank under w and under S1; a large displacement is one trigger of moderate discrepancy.

### Models

**Expected Solution Point (ESP)**:
The alternative the decision maker considers ideal; ESP-COMET and ESP-SPOTIS score alternatives by closeness to it.

**Score orientation**:
Whether a higher score means closer to the ESP (COMET preferences) or farther from it (SPOTIS distances).
_Avoid_: ascending, lower-is-better
