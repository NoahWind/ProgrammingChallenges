"""
play_stockfish.py
-----------------
Låter din schackbot spela ett parti mot Stockfish från en valfri FEN-ställning
och sparar ner partiet som en PGN-fil när det är färdigt.
"""

import datetime
import chess
import chess.engine
import chess.pgn
import torch

from nn_model import ChessEvaluatorNN, load_model
from search import find_best_move, clear_search_caches

# ==========================================
# GLOBALA VARIABLER / KONFIGURATION
# ==========================================
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Pekar direkt på din nedladdade Stockfish-fil i mappen
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe" 
MODEL_PATH = "chess_nnue.pth"

# Din bots inställningar
BOT_DEPTH = 7            # Sökdjup för din bot

# Stockfish inställningar
SF_TIME = 0.1            # Max tid per drag i sekunder (sätt till None om du inte vill ha tidsgräns)
SF_DEPTH = 4            # Max sökdjup för Stockfish (sätt till None om du bara vill köra på tid)

# PGN-inställningar
SAVE_PGN = True
PGN_FILENAME = "parti_resultat.pgn"
# ==========================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[bot] Laddar modell på {device}...")
    model = ChessEvaluatorNN().to(device)
    load_model(model, MODEL_PATH, device=device)
    model.eval()

    try:
        board = chess.Board(START_FEN)
    except ValueError as e:
        print(f"Ogiltig FEN-sträng: {e}")
        return

    print(f"\nStartar parti från FEN:\n{START_FEN}\n")

    # Skapa upp PGN-spelobjektet och sätt headers
    game = chess.pgn.Game()
    game.headers["Event"] = "Bot Test Match"
    game.headers["Site"] = "Local Python Environment"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "ChessNNUE Bot"
    game.headers["Black"] = f"Stockfish (Depth {SF_DEPTH})"
    if START_FEN != chess.STARTING_FEN:
        game.headers["FEN"] = START_FEN
        game.headers["SetUp"] = "1"

    node = game

    # Starta Stockfish-motorn med popen_uci
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Kunde inte starta Stockfish vid '{STOCKFISH_PATH}': {e}")
        print("Kontrollera att sökvägen till Stockfish är korrekt.")
        return

    history = set()

    try:
        while not board.is_game_over():
            print("-" * 40)
            print(board)
            print(f"Tur i draget: {'Vit' if board.turn == chess.WHITE else 'Svart'}")
            #stockfish analys av posetionen
            print("Stockfish analyserar positionen...")
            info = engine.analyse(board, chess.engine.Limit(time=SF_TIME, depth=SF_DEPTH))
            score = info["score"].white().score(mate_score=10000)
            print(f"Stockfish utvärdering: {score if score is not None else 'Mate'}")

            # Rensa sökkoder mellan dragen för att undvika gamla artefakter
            clear_search_caches()

            if board.turn == chess.WHITE:
                print("\nDin bot (Vit) funderar...")
                move, score = find_best_move(
                    board, model, depth=BOT_DEPTH, device=device, history=history
                )
                if move is None:
                    print("Botten hittade inga lagliga drag.")
                    break
                print(f"-> Bot spelar: {board.san(move)} (Uppskattad utvärdering: {score:.2f})")
            else:
                print("\nStockfish (Svart) funderar...")
                result = engine.play(board, chess.engine.Limit(time=SF_TIME, depth=SF_DEPTH))
                move = result.move
                print(f"-> Stockfish spelar: {board.san(move)}")

            # Lägg till draget i PGN-trädet
            node = node.add_variation(move)

            # Spara ställningsnyckel till historiken för remikontroll
            history.add(board.fen().split()[0])
            board.push(move)

        print("\n" + "=" * 40)
        print("PARTIET ÄR SLUT!")
        result_str = board.result()
        print(f"Resultat: {result_str}")
        print("Slutställning:")
        print(board)

        # Sätt slutresultatet i PGN-headern och spara till fil
        game.headers["Result"] = result_str
        if SAVE_PGN:
            with open(PGN_FILENAME, "w", encoding="utf-8") as pgn_file:
                print(game, file=pgn_file)
            print(f"Partiet har sparats till PGN-filen: {PGN_FILENAME}")

    finally:
        engine.quit()


if __name__ == "__main__":
    main()