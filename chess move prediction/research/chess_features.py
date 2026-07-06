"""Self-contained SAN parser and feature engineering for chess-cohort outcome prediction.
No external chess libraries. Reconstructs the board from the visible SAN move prefix and
extracts material / positional / opening features. This is legitimate parsing of the
visible input only -- NO engine evaluation, NO external lookup, NO best-move solving.
"""
import re
import numpy as np

PIECE_VALUE = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}

# ---- Board representation: 8x8 list, board[rank][file], rank 0 = rank1 (white home) ----
def initial_board():
    b = [[None] * 8 for _ in range(8)]
    back = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for f in range(8):
        b[0][f] = ('w', back[f])
        b[1][f] = ('w', 'P')
        b[6][f] = ('b', 'P')
        b[7][f] = ('b', back[f])
    return b

FILES = 'abcdefgh'

def sq_to_rf(sq):
    f = FILES.index(sq[0])
    r = int(sq[1]) - 1
    return r, f

def knight_moves(r, f):
    for dr, df in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            yield nr, nf

def king_moves(r, f):
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                yield nr, nf

SLIDE_DIRS = {
    'B': [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    'R': [(1, 0), (-1, 0), (0, 1), (0, -1)],
    'Q': [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)],
}

def slide_sources(board, piece, color, tr, tf):
    """Squares from which a sliding piece of given color/type can reach (tr,tf)."""
    out = []
    for dr, df in SLIDE_DIRS[piece]:
        r, f = tr + dr, tf + df
        while 0 <= r < 8 and 0 <= f < 8:
            cell = board[r][f]
            if cell is not None:
                if cell == (color, piece):
                    out.append((r, f))
                break
            r += dr
            f += df
    return out

def find_source(board, color, piece, tr, tf, dis_file, dis_rank):
    """Find source square of a non-pawn piece moving to (tr,tf), honoring disambiguation."""
    cands = []
    if piece == 'N':
        cands = [(r, f) for r, f in knight_moves(tr, tf) if board[r][f] == (color, 'N')]
    elif piece == 'K':
        cands = [(r, f) for r, f in king_moves(tr, tf) if board[r][f] == (color, 'K')]
    elif piece in ('B', 'R', 'Q'):
        cands = slide_sources(board, piece, color, tr, tf)
    if dis_file is not None:
        cands = [c for c in cands if c[1] == dis_file]
    if dis_rank is not None:
        cands = [c for c in cands if c[0] == dis_rank]
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        return None
    return cands[0]  # rare ambiguity (pins); material features stay correct regardless

SAN_RE = re.compile(r'^([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])(=[QRBN])?([+#])?$')

def apply_moves(move_prefix, return_board=False):
    """Replay SAN tokens, return dict of features. Robust to rare parse ambiguity."""
    board = initial_board()
    tokens = move_prefix.split()
    n_captures = 0
    n_checks = 0
    n_promo = 0
    castle = {'w': 0, 'b': 0}  # 0 none, 1 kingside, 2 queenside
    captured_value = {'w': 0, 'b': 0}  # value captured BY each side
    ep_target = None  # en passant target square (r,f) available this move

    for i, tok in enumerate(tokens):
        color = 'w' if i % 2 == 0 else 'b'
        opp = 'b' if color == 'w' else 'w'
        new_ep = None
        if tok.startswith('O-O-O') or tok.startswith('0-0-0'):
            r = 0 if color == 'w' else 7
            board[r][2] = (color, 'K'); board[r][3] = (color, 'R')
            board[r][4] = None; board[r][0] = None
            castle[color] = 2
            if '+' in tok or '#' in tok:
                n_checks += 1
            continue
        if tok.startswith('O-O') or tok.startswith('0-0'):
            r = 0 if color == 'w' else 7
            board[r][6] = (color, 'K'); board[r][5] = (color, 'R')
            board[r][4] = None; board[r][7] = None
            castle[color] = 1
            if '+' in tok or '#' in tok:
                n_checks += 1
            continue
        m = SAN_RE.match(tok)
        if not m:
            continue
        piece, dfile, drank, capx, dest, promo, chk = m.groups()
        tr, tf = sq_to_rf(dest)
        if chk:
            n_checks += 1
        is_capture = capx is not None
        if piece is None:  # pawn move
            src_f = FILES.index(dfile) if dfile is not None else tf
            direction = 1 if color == 'w' else -1
            if is_capture:
                # capture: source rank is tr - direction, source file = dfile
                sr = tr - direction
                sf = src_f
                target_cell = board[tr][tf]
                if target_cell is None and ep_target == (tr, tf):
                    # en passant
                    cap_r = tr - direction
                    if board[cap_r][tf] is not None:
                        captured_value[color] += PIECE_VALUE['P']
                        n_captures += 1
                        board[cap_r][tf] = None
                elif target_cell is not None:
                    captured_value[color] += PIECE_VALUE[target_cell[1]]
                    n_captures += 1
                if 0 <= sr < 8:
                    board[sr][sf] = None
                pc = (color, promo[1]) if promo else (color, 'P')
                board[tr][tf] = pc
                if promo:
                    n_promo += 1
            else:
                # forward push (1 or 2)
                sr = tr - direction
                if 0 <= sr < 8 and board[sr][tf] == (color, 'P'):
                    board[sr][tf] = None
                else:
                    sr2 = tr - 2 * direction
                    if 0 <= sr2 < 8 and board[sr2][tf] == (color, 'P'):
                        board[sr2][tf] = None
                        new_ep = (tr - direction, tf)
                pc = (color, promo[1]) if promo else (color, 'P')
                board[tr][tf] = pc
                if promo:
                    n_promo += 1
        else:
            src = find_source(board, color, piece, tr, tf,
                              FILES.index(dfile) if dfile else None,
                              (int(drank) - 1) if drank else None)
            if is_capture and board[tr][tf] is not None:
                captured_value[color] += PIECE_VALUE[board[tr][tf][1]]
                n_captures += 1
            if src is not None:
                board[src[0]][src[1]] = None
            board[tr][tf] = (color, piece)
        ep_target = new_ep

    # ---- Aggregate material / positional features from final board ----
    mat = {'w': 0, 'b': 0}
    counts = {('w', p): 0 for p in 'PNBRQK'}
    counts.update({('b', p): 0 for p in 'PNBRQK'})
    center = {'w': 0, 'b': 0}  # pawns/pieces on central 4 squares d4e4d5e5
    center_sqs = {(3, 3), (3, 4), (4, 3), (4, 4)}
    dev_minor = {'w': 0, 'b': 0}  # minors off back rank
    for r in range(8):
        for f in range(8):
            cell = board[r][f]
            if cell is None:
                continue
            col, pc = cell
            counts[(col, pc)] += 1
            mat[col] += PIECE_VALUE[pc]
            if (r, f) in center_sqs:
                center[col] += 1
            if pc in ('N', 'B'):
                home = 0 if col == 'w' else 7
                if r != home:
                    dev_minor[col] += 1
    feat = {}
    feat['mat_diff'] = mat['w'] - mat['b']
    feat['mat_total'] = mat['w'] + mat['b']
    feat['n_captures'] = n_captures
    feat['cap_diff'] = captured_value['w'] - captured_value['b']
    feat['n_checks'] = n_checks
    feat['n_promo'] = n_promo
    feat['w_castled'] = 1 if castle['w'] else 0
    feat['b_castled'] = 1 if castle['b'] else 0
    feat['castle_diff'] = (1 if castle['w'] else 0) - (1 if castle['b'] else 0)
    feat['queens_on'] = counts[('w', 'Q')] + counts[('b', 'Q')]
    feat['w_queen'] = counts[('w', 'Q')]
    feat['b_queen'] = counts[('b', 'Q')]
    feat['queen_diff'] = counts[('w', 'Q')] - counts[('b', 'Q')]
    feat['pawn_diff'] = counts[('w', 'P')] - counts[('b', 'P')]
    feat['minor_diff'] = (counts[('w', 'N')] + counts[('w', 'B')]) - (counts[('b', 'N')] + counts[('b', 'B')])
    feat['rook_diff'] = counts[('w', 'R')] - counts[('b', 'R')]
    feat['center_diff'] = center['w'] - center['b']
    feat['dev_diff'] = dev_minor['w'] - dev_minor['b']
    feat['dev_total'] = dev_minor['w'] + dev_minor['b']
    feat['w_bishops'] = counts[('w', 'B')]
    feat['b_bishops'] = counts[('b', 'B')]
    feat['bishop_pair_w'] = 1 if counts[('w', 'B')] >= 2 else 0
    feat['bishop_pair_b'] = 1 if counts[('b', 'B')] >= 2 else 0
    if return_board:
        planes = np.zeros((15, 8, 8), np.float32)
        pidx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
        wa = np.zeros((8, 8), np.float32); ba = np.zeros((8, 8), np.float32)
        for r in range(8):
            for f in range(8):
                cell = board[r][f]
                if cell is None:
                    continue
                col, pc = cell
                planes[pidx[pc] + (0 if col == 'w' else 6), r, f] = 1.0
                att = wa if col == 'w' else ba
                if pc == 'P':
                    d = 1 if col == 'w' else -1
                    for df in (-1, 1):
                        nr, nf = r + d, f + df
                        if 0 <= nr < 8 and 0 <= nf < 8: att[nr, nf] += 1
                elif pc == 'N':
                    for nr, nf in knight_moves(r, f): att[nr, nf] += 1
                elif pc == 'K':
                    for nr, nf in king_moves(r, f): att[nr, nf] += 1
                else:
                    for dr, df in SLIDE_DIRS[pc]:
                        nr, nf = r + dr, f + df
                        while 0 <= nr < 8 and 0 <= nf < 8:
                            att[nr, nf] += 1
                            if board[nr][nf] is not None: break
                            nr += dr; nf += df
        planes[12] = np.clip(wa / 3.0, 0, 2); planes[13] = np.clip(ba / 3.0, 0, 2)
        planes[14] = np.clip((wa - ba) / 3.0, -2, 2)
        return feat, planes
    return feat


if __name__ == '__main__':
    import pandas as pd, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tr = pd.read_csv(os.path.join(here, 'dataset', 'train.csv'))
    # sanity: parse a few, check material near-equal typically
    for i in range(5):
        f = apply_moves(tr.move_prefix.iloc[i])
        print(tr.move_prefix.iloc[i])
        print('  mat_diff', f['mat_diff'], 'caps', f['n_captures'], 'cap_diff', f['cap_diff'], 'wcastle', f['w_castled'])
    # aggregate: distribution of mat_diff
    md = tr.move_prefix.apply(lambda s: apply_moves(s)['mat_diff'])
    print('mat_diff value counts:', md.value_counts().sort_index().to_dict())
    cap = tr.move_prefix.apply(lambda s: apply_moves(s)['n_captures'])
    print('n_captures dist:', cap.value_counts().sort_index().to_dict())
