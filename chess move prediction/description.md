Overview
You are given early-to-midgame chess move prefixes as SAN token strings. Each row represents a real aggregate cohort of games that share the visible prefix pattern, not a single game. This is an NLP-style symbolic sequence modeling task: map a SAN token sequence to a calibrated three-class outcome distribution for the hidden cohort: white win, draw, and black win.

The public rows contain only an opaque id, the SAN move_prefix, prefix_ply_count, and side_to_move. They do not contain player names, game IDs, dates, source filenames, ratings, event/site/round tags, ECO/opening names, termination, result tags, raw order, or full continuations. Training rows include empirical outcome rates for aggregate cohorts; test rows hide those rates in private/answers.csv.

What to use: fine-tune a compact text/sequence encoder on the public rows, train a SAN-token model from scratch, build n-gram or transformer-style language features, fit calibrated probabilistic classifiers/regressors, and validate calibration on the public training labels. What not to use: external PGN lookup, engine evaluation, best-move solving, source archive reconstruction, row-order side channels, or grader exploitation. The intended challenge is sequence-to-distribution learning from the visible prefix only.

Evaluation
Submissions provide four values per test row: three class probabilities and a confidence score. The grader first validates the exact schema and ID set. Structural CSV errors return 0.0; malformed row-local prediction values give only that row zero contribution rather than crashing or zeroing the whole file.

For valid submissions, the grader computes a composite probability skill score:

S_brier = clip(1 - mean_valid_rows(sum((p - y)^2)) / BRIER_REF, 0, 1)  
S_log   = clip(1 - mean_valid_rows(CE(y, p) - CE(y, y)) / LOG_REF, 0, 1)  
S_conf  = clip(1 - mean_valid_rows(abs(confidence - max(y))) / CONF_REF, 0, 1)  
S_worst = lowest hidden-group Brier skill, with malformed rows counted as zero rows in their group  
Composite = 0.55*S_brier + 0.20*S_log + 0.10*S_conf + 0.15*S_worst  
Final = valid_row_fraction * (0.12 + 0.88*Composite)  

y is the hidden empirical outcome distribution. p is the submitted distribution after row-wise normalization. BRIER_REF, LOG_REF, and CONF_REF are fixed constants derived from the prepared training split, not hidden test marginals. The valid-row fraction makes malformed rows contribute zero while preserving smooth scoring for the remaining rows. Structurally invalid submissions still score exactly 0.0. A perfect submission that predicts the hidden distribution and sets confidence to the largest hidden class probability scores exactly 1.0.

Higher is better. Theoretical minimum: 0.0. Theoretical maximum: 1.0.

Dataset
Participants receive public/train.csv, public/test.csv, and public/sample_submission.csv. public/train.csv contains one row per labeled aggregate prefix cohort, including the empirical target rates and the hidden cohort size for training only. public/test.csv contains the same visible prefix fields without target rates or cohort size. public/sample_submission.csv is a valid weak template based on public training priors.

The visible test input columns are id, move_prefix, prefix_ply_count, and side_to_move. The train-only label columns are white_win_rate, draw_rate, black_win_rate, and cohort_game_count; the three rate columns are the empirical cohort target distribution.

File overview
Item	Description
public/train.csv	Labeled prefix cohorts
public/test.csv	Unlabeled prefix cohorts
public/sample_submission.csv	Valid weak template
private/answers.csv	Hidden target rates
Train columns
Column	Type	Description
id	int	Opaque row id
move_prefix	string	SAN tokens
prefix_ply_count	int	Prefix length
side_to_move	string	Next side
white_win_rate	float	Cohort white-win rate
draw_rate	float	Cohort draw rate
black_win_rate	float	Cohort black-win rate
cohort_game_count	int	Train cohort size
The three rate columns are nonnegative and sum to 1. cohort_game_count is shown only for training rows so solvers can learn how empirical cohort size relates to calibration.

Test columns
Column	Type	Description
id	int	Opaque row id
move_prefix	string	SAN tokens
prefix_ply_count	int	Prefix length
side_to_move	string	Next side
Test rows hide all target rates, cohort sizes, source identifiers, and continuations. id and row order are salted and shuffled; they carry no source-order or split signal.

Submission format
Column	Type	Constraint
id	int	Match test ids
white_win_prob	float	[0, 1]
draw_prob	float	[0, 1]
black_win_prob	float	[0, 1]
confidence	float	[0, 1]
The three probability columns should be finite floats in [0, 1]; valid nonzero rows are normalized before scoring. A row with malformed, non-finite, out-of-range, or all-zero probability values receives zero row-level contribution for that row only. confidence should be finite and in [0, 1]; malformed, non-finite, or out-of-range confidence also makes only that row contribute zero.

Submission
Submit a CSV with exactly these columns in this order: id, white_win_prob, draw_prob, black_win_prob, confidence.

Example:

id,white_win_prob,draw_prob,black_win_prob,confidence  
105,0.52,0.05,0.43,0.52  
418,0.45,0.08,0.47,0.47  
912,0.61,0.03,0.36,0.61  

The full submission must contain exactly one row for every id in public/test.csv. Duplicate IDs, missing IDs, extra IDs, missing columns, or extra/reordered columns return 0.0. Malformed probability or confidence cells are not useful, but they zero only the affected row.

What Not To Do
Using any of these approaches is grounds for solution rejection on review, regardless of leaderboard score:

External PGN or game-database lookup to reconstruct hidden test cohorts, source rows, continuations, or outcomes.
Chess-engine evaluation, Stockfish labels, best-move prediction, puzzle solving, or engine-assisted position scoring.
Full-game result lookup, opening-book table lookup, or exact source archive counting instead of learning from public/train.csv.
Player/rating/date/source metadata reconstruction or exploiting row order, file metadata, salted ID patterns, or source import details.
Hosted or closed-source APIs for training, inference, pseudo-labeling, or distillation.
Grader or platform exploitation, including malformed probability rows, hidden-file probing, or hard-coded answer dictionaries.
Enforcement on Invalid Approaches
Solutions that rely on source lookup, engine solving, hard-coded source reconstruction, hidden-file access, or format exploitation may be rejected before payout even if the submitted CSV receives a high score. Valid approaches must train or fit from the public training file and make predictions from the visible prefix fields only.

 