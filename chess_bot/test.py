import os
import chess
import chess.engine
import chess.pgn
import pandas as pd

# Konfiguration
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
OUTPUT_DIR = "stockfish_self_play_results"
TIME_LIMIT = 0.7  # Tidsgräns per drag i sekunder

pgn_file_path = os.path.join(OUTPUT_DIR, "games.pgn")
csv_file_path = os.path.join(OUTPUT_DIR, "positions_ratings.csv")

def process_existing_games():
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Fel: Kunde inte hitta {STOCKFISH_PATH}. Kontrollera att filen ligger i samma mapp.")
        return
        
    if not os.path.exists(pgn_file_path):
        print(f"Fel: Kunde inte hitta PGN-filen {pgn_file_path}.")
        return

    print(f"Läser och analyserar befintliga partier från {pgn_file_path}...")

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        with open(pgn_file_path, "r", encoding="utf-8") as pgn_file:
            game_idx = 0
            
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_idx += 1
                board = game.board()
                game_positions = []
                move_count = 0

                # Gå igenom varje drag i partiet
                for node in game.mainline():
                    move = node.move
                    current_fen = board.fen()
                    turn_color = "white" if board.turn == chess.WHITE else "black"

                    # 1. Analysera ställningen innan draget utförs
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

                    # Samla positionen
                    game_positions.append({
                        "game_id": game_idx,
                        "move_number": move_count + 1,
                        "fen": current_fen,
                        "move_uci": move.uci(),
                        "eval_cp": eval_cp,
                        "turn": turn_color
                    })

                    # Utför draget på brädet för nästa iteration
                    board.push(move)
                    move_count += 1

                # 2. SPARA till CSV direkt efter varje parti (med append-läge "a")
                df_game = pd.DataFrame(game_positions)
                if not os.path.exists(csv_file_path):
                    df_game.to_csv(csv_file_path, index=False, encoding="utf-8")
                else:
                    df_game.to_csv(csv_file_path, mode="a", header=False, index=False, encoding="utf-8")

                print(f"Parti {game_idx} färdiganalyserat och tillagt i CSV. Antal drag: {move_count}")

    print(f"\nThats a bingo! Alla sparade partier har bearbetats.")
    print(f" - CSV-fil uppdaterad: {csv_file_path}")

if __name__ == "__main__":
    process_existing_games()