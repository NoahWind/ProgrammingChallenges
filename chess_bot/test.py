import os
import time
import chess
import chess.engine
import chess.pgn
import pandas as pd

# Konfiguration
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
NUM_GAMES = 100
OUTPUT_DIR = "stockfish_self_play_results"
TIME_LIMIT = 0.1  # Tidsgräns per drag i sekunder

os.makedirs(OUTPUT_DIR, exist_ok=True)
pgn_file_path = os.path.join(OUTPUT_DIR, "games.pgn")
csv_file_path = os.path.join(OUTPUT_DIR, "positions_ratings.csv")

def run_simulation():
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Fel: Kunde inte hitta {STOCKFISH_PATH}. Kontrollera att filen ligger i samma mapp.")
        return

    print(f"Startar {NUM_GAMES} matcher med Stockfish självspel (sparar efter varje parti)...")

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        
        for game_idx in range(1, NUM_GAMES + 1):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = f"Stockfish Self-Play Match #{game_idx}"
            game.headers["Site"] = "Local"
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            game.headers["Round"] = str(game_idx)
            game.headers["White"] = "Stockfish AVX2"
            game.headers["Black"] = "Stockfish AVX2"

            node = game
            move_count = 0
            game_positions = []

            while not board.is_game_over() and move_count < 150:
                current_fen = board.fen()
                turn_color = "white" if board.turn == chess.WHITE else "black"

                # 1. Analysera ställningen innan draget görs
                info = engine.analyse(board, chess.engine.Limit(time=TIME_LIMIT))
                score = info.get("score")
                
                eval_cp = 0
                if score:
                    pov_score = score.pov(board.turn)
                    if pov_score.is_mate():
                        mate_in = pov_score.mate()
                        eval_cp = 10000 if mate_in > 0 else -10000
                    else:
                        eval_cp = pov_score.score(mate_score=10000)

                # 2. Låt Stockfish välja draget
                result = engine.play(board, chess.engine.Limit(time=TIME_LIMIT))
                move = result.move

                if move is None:
                    break

                # Samla positionen för detta parti
                game_positions.append({
                    "game_id": game_idx,
                    "move_number": move_count + 1,
                    "fen": current_fen,
                    "move_uci": move.uci(),
                    "eval_cp": eval_cp,
                    "turn": turn_color
                })

                # Utför draget
                board.push(move)
                node = node.add_variation(move)
                move_count += 1

            # Sätt slutresultatet i PGN-huvudet
            outcome = board.outcome()
            if outcome:
                game.headers["Result"] = outcome.result()
            else:
                game.headers["Result"] = "1/2-1/2"

            # 3. SPARA PGN direkt efter varje parti (append-läge "a")
            with open(pgn_file_path, "a", encoding="utf-8") as pgn_out:
                print(game, file=pgn_out, end="\n\n")

            # 4. SPARA CSV direkt efter varje parti (lägg till i befintlig fil eller skapa ny)
            df_game = pd.DataFrame(game_positions)
            if not os.path.exists(csv_file_path):
                df_game.to_csv(csv_file_path, index=False, encoding="utf-8")
            else:
                df_game.to_csv(csv_file_path, mode="a", header=False, index=False, encoding="utf-8")

            print(f"Match {game_idx}/{NUM_GAMES} klar och sparad. Drag: {move_count}, Resultat: {game.headers['Result']}")

    print(f"\nThats a bingo! Alla matcher slutförda.")
    print(f" - PGN sparad i: {pgn_file_path}")
    print(f" - CSV sparad i: {csv_file_path}")

if __name__ == "__main__":
    run_simulation()