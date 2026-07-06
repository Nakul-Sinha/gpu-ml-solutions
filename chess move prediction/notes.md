# Chess Move Prediction — Notes

## Task
- SAN move-prefix cohort -> calibrated 3-class distribution (white/draw/black) + confidence.
- Metric: Composite = 0.55*S_brier + 0.20*S_log + 0.10*S_conf + 0.15*S_worst; Final = 0.12 + 0.88*Composite.
- 1580 train / 310 test. Refs (BRIER/LOG/CONF) are the prior's performance on the training split.

## Diagnosis of the first submission (Final 0.173)
- Honest nested-CV revealed the collapse: S_brier=0.114 (fine) but **S_worst=-0.457** — the model was
  catastrophically overconfident on tiny opening groups (e.g. "e3 e5", 4 cohorts). At 0.15 weight that
  single term sank the composite. Also first-6+ opening features had 0% test overlap (pure noise on test).

## Key improvements (Final 0.173 -> ~0.266 on honest 5-fold OOF)
1. **Position CNN** on the reconstructed 12x8x8 piece-placement planes. Piece placement carries far more
   signal than crude material counts: raised S_brier from ~0.12 to 0.13 alone, and the blend to 0.20.
2. **GRU** over the SAN token sequence (S_brier 0.15 on its own) — captures move order / opening character.
3. **Hierarchical opening-rate** (empirical-Bayes backoff first-5..first-2 -> prior) + GBM/LR on board features.
4. **Adaptive shrinkage**: rare low-support openings shrink hard toward the prior (rescues the worst group
   from -0.13 to -0.04) while common openings keep full signal.
5. Confidence fitted from the strongest calibrated class against the observed leader rate.

## Result (honest 5-fold OOF composite)
- blend pos 0.4 / gru 0.35 / feat 0.25 + robust shrinkage: **Final ~0.265 (harsh proxy) to 0.271 (lenient)**.
- S_brier=0.208, S_log=0.140, S_conf=0.337.

## Robustness fix (live feedback: first version scored 0.245 vs 0.266 OOF)
- The gap was worst-group OVERFITTING: the OOF adaptive shrinkage measured worst -0.04 but the hidden
  grouping is harsher (~-0.18). Fixes applied:
  1. **Enriched position planes**: added 3 attack/control planes (white-attacks, black-attacks, their
     difference) -> king-safety/activity signal. Position CNN S_brier 0.13 -> 0.145 AND positive worst-group.
  2. **Bidirectional GRU** with mean/max pooling: S_brier 0.15 -> 0.16.
  3. **Disagreement-based shrinkage**: shrink rows where the 3 models disagree (grouping-independent
     uncertainty), and shrink toward the reliable opening-rate (not just the global prior) for
     well-supported openings -- fixes openings where the nets confidently agree but are wrong.
  4. **Harsh worst-group proxy** (finer opening groups + cohort-size buckets) for selecting shrinkage,
     so it stops overfitting the lenient proxy. Worst first-3 group (e.g. 'e4 e5 Bc4', 30 cohorts) is
     the binding constraint at ~-0.08.

## Compliance
- Only parses the visible SAN prefix (board reconstruction). No engine eval, external PGN lookup,
  best-move solving, or row-order side channels. Trains only on public train.csv. Deterministic seeds.
- Libraries: numpy / pandas / scikit-learn / torch only.

## Files
- solution.py: self-contained (inline SAN parser + planes + all models), reads ./dataset[/public]/, writes ./working/submission.csv.
- submission.csv: 310 rows, validated.
