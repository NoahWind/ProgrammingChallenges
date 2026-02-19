import os
import multiprocessing
from tqdm import tqdm

POINTS = {'a': 1, 'b': 3, 'c': 8, 'd': 1, 'e': 1, 'f': 3, 'g': 2, 'h': 3, 'i': 1, 'j': 7, 'k': 3, 'l': 2, 'm': 3, 'n': 1, 'o': 2, 'p': 4, 'r': 1, 's': 1, 't': 1, 'u': 4, 'v': 3, 'x': 8, 'y': 7, 'z': 10, 'å': 4, 'ä': 4, 'ö': 4}

LAYOUT = [
    [2, 0, 0, 0, 4, 0, 0, 1, 0, 0, 4, 0, 0, 0, 2],
    [0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0],
    [0, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0],
    [4, 0, 0, 0, 3, 0, 1, 0, 1, 0, 3, 0, 0, 0, 4],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0],
    [4, 0, 0, 0, 3, 0, 1, 0, 1, 0, 3, 0, 0, 0, 4],
    [0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 0, 3, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0],
    [2, 0, 0, 0, 4, 0, 0, 1, 0, 0, 4, 0, 0, 0, 2]
]

BMAP = {0: "..", 1: "DL", 2: "TL", 3: "DW", 4: "TW", 5: "★ "}

def load_real_board(fn="real_board.txt"):
    bd = []
    if not os.path.exists(fn): return None
    with open(fn, "r", encoding="utf-8") as f:
        for ln in f:
            c = [x.strip().lower() for x in ln.split(',')]
            bd.append([x if x.isalpha() else None for x in c])
    return bd

def calculate_score(w, r, c, d, bd):
    s = 0
    wm = 1
    for i, l in enumerate(w):
        curr_r = r + (i if d == 'v' else 0)
        curr_c = c + (i if d == 'h' else 0)
        bp = POINTS.get(l, 0)
        if bd[curr_r][curr_c] is None:
            bn = LAYOUT[curr_r][curr_c]
            if bn == 1: bp *= 2
            elif bn == 2: bp *= 3
            elif bn == 3: wm *= 2
            elif bn == 4: wm *= 3
        s += bp
    return s * wm

def print_visual_board(m, obd):
    print(f"\nDRAG: {m['word'].upper()} ({m['score']}p)")
    print("    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14")
    print("  " + "-" * 46)
    for r in range(15):
        row_s = f"{r:2}|"
        for c in range(15):
            nw = False
            if m['dir'] == 'h' and r == m['row'] and m['col'] <= c < m['col'] + len(m['word']):
                if obd[r][c] is None: nw = True
            elif m['dir'] == 'v' and c == m['col'] and m['row'] <= r < m['row'] + len(m['word']):
                if obd[r][c] is None: nw = True
            if nw:
                idx = (c - m['col']) if m['dir'] == 'h' else (r - m['row'])
                row_s += f"({m['word'][idx].upper()})"
            elif obd[r][c]:
                row_s += f" {obd[r][c].upper()} "
            else:
                row_s += f"{BMAP[LAYOUT[r][c]]}"
        print(row_s)

def get_full_word_at(bd, r, c, d):
    w = ""
    cr, cc = r, c
    while cr >= 0 and cc >= 0 and bd[cr][cc]:
        if d == 'v': cr -= 1
        else: cc -= 1
    if d == 'v': cr += 1
    else: cc += 1
    while cr < 15 and cc < 15 and bd[cr][cc]:
        w += bd[cr][cc]
        if d == 'v': cr += 1
        else: cc += 1
    return w if len(w) > 1 else None

def check_row_worker(args):
    r, rack, bd, d_set = args
    res = []
    for c in range(15):
        for d in ['h', 'v']:
            for w in d_set:
                if (d == 'h' and c + len(w) > 15) or (d == 'v' and r + len(w) > 15): continue
                tr, ok, conn = list(rack), True, False
                tbd = [row[:] for row in bd]
                coords = []
                for i, char in enumerate(w):
                    cr, cc = r + (i if d == 'v' else 0), c + (i if d == 'h' else 0)
                    if bd[cr][cc]:
                        if bd[cr][cc] != char: ok = False; break
                        conn = True
                    else:
                        if char in tr:
                            tr.remove(char)
                            tbd[cr][cc] = char
                            coords.append((cr, cc))
                        else:
                            ok = False; break
                if not ok: continue
                mw = get_full_word_at(tbd, r, c, d)
                if mw != w or mw not in d_set:
                    if mw not in d_set: continue
                if len(mw) > len(w): conn = True
                pd = 'v' if d == 'h' else 'h'
                for pr, pc in coords:
                    cw = get_full_word_at(tbd, pr, pc, pd)
                    if cw:
                        if cw not in d_set: ok = False; break
                        conn = True
                if ok and conn:
                    sc = calculate_score(w, r, c, d, bd)
                    res.append({'word': w, 'row': r, 'col': c, 'dir': d, 'score': sc})
    return res

if __name__ == "__main__":
    try:
        with open("wordfeud_master.txt", "r", encoding="utf-8") as f:
            v_words = {line.strip().lower() for line in f if len(line.strip()) > 1}
    except:
        exit()

    board = load_real_board()
    if board:
        my_rack = input("Bokstäver: ").lower()
        args = [(r, my_rack, board, v_words) for r in range(15)]
        found = []
        with multiprocessing.Pool() as pool:
            for r_moves in tqdm(pool.imap_unordered(check_row_worker, args), total=15):
                found.extend(r_moves)
        
        uniq = {f"{m['word']}_{m['row']}_{m['col']}_{m['dir']}": m for m in found}
        final = sorted(uniq.values(), key=lambda x: x['score'], reverse=True)[:5]
        for m in final:
            print_visual_board(m, board)