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
- blend pos 0.35 / gru 0.35 / feat 0.30 + adaptive shrink: composite=0.166, **Final ~0.266**
- S_brier=0.202, S_log=0.135, S_conf=0.338, S_worst=-0.039.

## Compliance
- Only parses the visible SAN prefix (board reconstruction). No engine eval, external PGN lookup,
  best-move solving, or row-order side channels. Trains only on public train.csv. Deterministic seeds.
- Libraries: numpy / pandas / scikit-learn / torch only.

## Files
- solution.py: self-contained (inline SAN parser + planes + all models), reads ./dataset[/public]/, writes ./working/submission.csv.
- submission.csv: 310 rows, validated.
