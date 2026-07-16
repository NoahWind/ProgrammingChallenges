import bitbully as bb
import json
import os
import sqlite3
import time

WIDTH = 7
HEIGHT = 6
CELL_BITS = 2
TOKEN_CODE = {'x': 1, 'o': 2}

SQLITE_FILE = "connect4_perfect_db.sqlite3"
JSON_FILE = "connect4_perfect_db.json"

COMMIT_EVERY = 2000
PRINT_EVERY = 20

POSITIONS_BY_DEPTH = {
    0: 1, 1: 8, 2: 57, 3: 295, 4: 1415, 5: 5678, 6: 22100, 7: 76959,
    8: 261234, 9: 819420, 10: 2482043, 11: 7050726, 12: 19286827,
    13: 50215938, 14: 125653533, 15: 302194792,
}


def init_db():
    conn = sqlite3.connect(SQLITE_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Om en äldre version av databasen finns kvar med fel kolumntyp
    # (t.ex. INTEGER istället för TEXT för nyckeln), återskapa tabellen.
    cols = conn.execute("PRAGMA table_info(positions)").fetchall()
    if cols and cols[0][2].upper() != "TEXT":
        conn.execute("DROP TABLE positions")
        conn.commit()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            key TEXT PRIMARY KEY,
            score INTEGER,
            col INTEGER,
            row INTEGER
        )
    """)
    conn.commit()
    return conn


def flush(conn, pending):
    if pending:
        conn.executemany(
            "INSERT OR REPLACE INTO positions (key, score, col, row) VALUES (?, ?, ?, ?)",
            pending
        )
        conn.commit()
        pending.clear()


def decode_key(key):
    alternating = bool(key & 1)
    board_code = key >> 1
    board = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for col in range(WIDTH):
        for row in range(HEIGHT):
            shift = (col * HEIGHT + row) * CELL_BITS
            val = (board_code >> shift) & 0b11
            if val == 1:
                board[row][col] = 'x'
            elif val == 2:
                board[row][col] = 'o'
    return board, alternating


def export_json():
    conn = sqlite3.connect(SQLITE_FILE)
    result = {}
    for key_str, score, col, row in conn.execute("SELECT key, score, col, row FROM positions"):
        board, alternating = decode_key(int(key_str))
        json_key = json.dumps([[list(r) for r in board], alternating])
        result[json_key] = [score, col, row]
    conn.close()
    with open(JSON_FILE, "w") as f:
        json.dump(result, f)


def build_dataset(depth_limit=6):
    conn = init_db()
    known = set(int(r[0]) for r in conn.execute("SELECT key FROM positions"))
    visited = set()
    pending = []
    agent = bb.BitBully()

    heights = [0] * WIDTH
    move_seq = []
    state = {"board_code": 0, "new_count": 0}
    total_expected = POSITIONS_BY_DEPTH.get(depth_limit)
    start_time = time.time()

    def explore(alternating):
        depth = len(move_seq)
        if depth > depth_limit:
            return

        key = (state["board_code"] << 1) | (1 if alternating else 0)

        if key not in known:
            bb_board = bb.Board(move_seq) if move_seq else bb.Board()

            if bb_board.is_game_over():
                score, best_col, row = 0, None, None
            else:
                raw_score = agent.mtdf(bb_board)
                score = 1 if raw_score > 0 else (-1 if raw_score < 0 else 0)
                best_col = agent.best_move(bb_board)
                row = heights[best_col]

            pending.append((str(key), score, best_col, row))
            known.add(key)
            state["new_count"] += 1

            if len(pending) >= COMMIT_EVERY:
                flush(conn, pending)

            if state["new_count"] % PRINT_EVERY == 0:
                if total_expected:
                    pct = min(len(known) / total_expected, 1.0)
                    bar = "#" * int(pct * 30)
                    elapsed = time.time() - start_time
                    rate = len(known) / elapsed if elapsed > 0 else 0
                    eta = (total_expected - len(known)) / rate if rate > 0 else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                    print(f"\r[{bar:<30}] {pct*100:5.1f}% ({len(known)}/{total_expected}) | kvar: {eta_str}", end="", flush=True)
                else:
                    print(f"  ...{state['new_count']} nya positioner sparade hittills")

        if key in visited:
            return
        visited.add(key)

        if depth < depth_limit:
            turn = 'x' if alternating else 'o'
            token_code = TOKEN_CODE[turn]
            for col in range(WIDTH):
                if heights[col] < HEIGHT:
                    row = heights[col]
                    shift = (col * HEIGHT + row) * CELL_BITS

                    state["board_code"] |= token_code << shift
                    heights[col] += 1
                    move_seq.append(col)

                    explore(not alternating)

                    move_seq.pop()
                    heights[col] -= 1
                    state["board_code"] ^= token_code << shift

    explore(True)
    flush(conn, pending)
    conn.close()

    print()
    print(f"Beräkning klar, {state['new_count']} nya positioner. Exporterar till JSON...")
    export_json()
    print("Dataset klart!")


if __name__ == "__main__":
    start = time.time()
    print("Startar nu klockan", time.strftime("%H:%M:%S", time.localtime(start)))
    depth = 13
    print(f"Bygger dataset med djup {depth}...")
    build_dataset(depth_limit=depth)
    end = time.time()
    print(f"Tid: {end - start:.2f} sekunder.")