# Chess Move Prediction — Notes

## Task
- Map SAN move-prefix (aggregate cohort) -> 3-class outcome distribution (white/draw/black) + confidence.
- Metric: composite skill = 0.55*S_brier + 0.20*S_log + 0.10*S_conf + 0.15*S_worst. Higher better, [0,1].
- 1580 train cohorts, 310 test. side_to_move always "white". Draws rare (mean 3.7%, 48% of cohorts have 0 draws).
- 0/310 test prefixes appear exactly in train -> must generalize from move patterns, not memorize.

## Key findings
- Full positional one-hot features OVERFIT (BSS -0.11 at C=1). Only generalizable features work.
- Board material/tactical features (from SAN replay) + opening-family target-encoding are the signal.
- GBM (heavy reg: 8 leaves, min_leaf 50) > LR. Blend GBM 0.8 / LR 0.2.
- Prior shrinkage lam=0.25 + fitted confidence map both help.

## Result (repeated 5-fold x3 CV)
- OOF Brier Skill Score = 0.1576 (vs weighted prior).
- log-excess (KL) = 0.077.
- Confidence MAE 0.078 (vs 0.097 raw max-prob).
- Group BSS: e4=0.107, d4=0.297 (test is only e4/d4 openings).

## Compliance
- Only parses the visible SAN prefix (board reconstruction). No engine eval, no external PGN lookup,
  no best-move solving, no row-order/id side channels, no hardcoded answers.
- Trains only on public/train.csv. Deterministic. Standard libs only (numpy/pandas/sklearn).

## Files
- solution.py: self-contained, reads ./dataset[/public]/, writes ./working/submission.csv.
- submission.csv: 310 rows, validated (schema, normalized probs, finite confidence).
