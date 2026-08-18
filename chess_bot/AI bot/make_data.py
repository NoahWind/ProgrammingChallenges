"""
create_mega_db.py
-----------------
1. Streamar data från Lichess parquet-databasen.
2. Filtrerar med pre-filtret för rena 'one-move blunders' (lagliga fångster av oskyddade/underdefenderade pjäser).
3. Använder multiprocessing och Stockfish för att snabbt sätta exakt cp/mate.
4. Sparar kontinuerligt till 'min_schack_databas/mega_hanging_evals.parquet'.
5. Sparar bildbevis på senaste träffen.
"""

import os
import multiprocessing
import chess
import chess.engine
import chess.svg
import pyarrow.dataset as ds
import pandas as pd

STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
DB_PATH = "min_schack_databas/lichess_data.parquet"
OUTPUT_DB = "min_schack_databas/mega_hanging_evals.parquet"
LATEST_IMG_PATH = "min_schack_databas/latest_batch.svg"

MAX_TARGET = 100_000
NODES_LIMIT = 5_000      # Optimerad för maximal hastighet per ställning
CHECKPOINT_EVERY = 1_000
NUM_WORKERS = 2          # Sätt till antalet kärnor du vill använda (t.ex. 4)

_engine = None

def init_worker(stockfish_path: str):
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _engine.configure({"Threads": 1, "Hash": 16})

def is_clean_position(board: chess.Board) -> bool:
    """Rensar bort trasiga eller orealistiska ställningar."""
    if len(board.pieces(chess.KING, chess.WHITE)) != 1 or len(board.pieces(chess.KING, chess.BLACK)) != 1:
        return False
    try:
        if not board.is_valid():
            return False
    except Exception:
        return False
    return True

def has_one_move_hanging_tactic(board: chess.Board) -> bool:
    """
    Super-effektiv pre-filter för one-move blunders:
    Kollar enbart på spelarens lagliga fångst-drag. Om ett drag slår 
    en oskyddad eller underdefenderad pjäs direkt, flaggar vi ställningen!
    """
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
    opponent_color = not board.turn
    
    for move in board.legal_moves:
        if board.is_capture(move):
            target_square = move.to_square
            target_piece = board.piece_at(target_square)
            
            if not target_piece or target_piece.piece_type == chess.KING:
                continue
            
            defenders = board.attackers(opponent_color, target_square)
            
            # 1. Helt oskyddad pjäs (Gratis one-move blunder)
            if len(defenders) == 0:
                return True
            
            attacker_piece = board.piece_at(move.from_square)
            if not attacker_piece:
                continue
            
            attacker_val = piece_values[attacker_piece.piece_type]
            target_val = piece_values[target_piece.piece_type]
            
            attackers = board.attackers(board.turn, target_square)
            # 2. Fler attackerare än försvarare och vi slår med en billigare eller lika värd pjäs
            if len(attackers) > len(defenders) and attacker_val <= target_val:
                return True
                
    return False

def process_fen(fen: str):
    global _engine
    try:
        board = chess.Board(fen)
    except Exception:
        return None

    # Snabbkoll via vårt pre-filter
    if not is_clean_position(board) or not has_one_move_hanging_tactic(board):
        return None

    try:
        info = _engine.analyse(board, chess.engine.Limit(nodes=NODES_LIMIT))
        score = info["score"].white()
        if score.is_mate():
            return (fen, None, score.mate())
        return (fen, score.score(mate_score=10000), None)
    except Exception:
        return None

def save_checkpoint_append(batch_rows, output_path):
    if not batch_rows:
        return
    df_new = pd.DataFrame(batch_rows)
    if os.path.exists(output_path):
        df_old = pd.read_parquet(output_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    temp_path = output_path + ".tmp"
    df_combined.to_parquet(temp_path)
    os.replace(temp_path, output_path)

def save_latest_image(fen, filepath):
    """Skapar en SVG-bild av ställningen och skriver över den gamla."""
    try:
        board = chess.Board(fen)
        svg_data = chess.svg.board(board, size=400)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_data)
    except Exception:
        pass

def fen_stream(db_path: str):
    dataset = ds.dataset(db_path)
    scanner = dataset.scanner(columns=["fen"])
    for batch in scanner.to_batches():
        for fen in batch.to_pydict()["fen"]:
            yield fen

def main():
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Hittar inte Stockfish på '{STOCKFISH_PATH}'.")
        return

    os.makedirs(os.path.dirname(OUTPUT_DB), exist_ok=True)

    seen_fens = set()
    total_saved = 0

    if os.path.exists(OUTPUT_DB):
        print(f"Hittade befintlig databas på '{OUTPUT_DB}'. Läser in...")
        df_existing = pd.read_parquet(OUTPUT_DB)
        seen_fens.update(df_existing["fen"].tolist())
        total_saved = len(seen_fens)
        print(f"Återupptar arbetet. Har redan {total_saved} ställningar klara.")

    if total_saved >= MAX_TARGET:
        print("Målet är redan uppnått!")
        return

    print(f"Startar {NUM_WORKERS} parallella Stockfish-workers med one-move blunder-filter (mål: {MAX_TARGET})...")

    batch_results = []
    pool = multiprocessing.Pool(NUM_WORKERS, initializer=init_worker, initargs=(STOCKFISH_PATH,))

    try:
        for res in pool.imap_unordered(process_fen, fen_stream(DB_PATH), chunksize=200):
            if res is None:
                continue

            fen, cp, mate = res
            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            batch_results.append({"fen": fen, "cp": cp, "mate": mate})
            total_saved += 1

            if total_saved % 1000 == 0:
                print(f"Hittade och utvärderade {total_saved} / {MAX_TARGET} ställningar...")

            if len(batch_results) >= CHECKPOINT_EVERY:
                save_checkpoint_append(batch_results, OUTPUT_DB)
                save_latest_image(batch_results[-1]["fen"], LATEST_IMG_PATH)
                print(f"--> Checkpoint sparad. Tömmer batch ur minnet.")
                batch_results.clear()

            if total_saved >= MAX_TARGET:
                break
    except KeyboardInterrupt:
        print("\nAvbrutet av användaren - sparar sista batchen...")
    finally:
        pool.terminate()
        pool.join()

    if batch_results:
        save_checkpoint_append(batch_results, OUTPUT_DB)
        save_latest_image(batch_results[-1]["fen"], LATEST_IMG_PATH)
        
    print(f"\nKlart! Databasen '{OUTPUT_DB}' innehåller nu {total_saved} ställningar.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()