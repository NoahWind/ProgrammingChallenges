"""
check_hang_recognition.py
--------------------------
Testar om DIN tränade modell (inte sökningen) förstår att en hängande pjäs är dåligt.
Kör lokalt med din riktiga chess_nnue.pth. Kräver ingen Stockfish.

Kör:
    python check_hang_recognition.py --model-path chess_nnue.pth
"""
import argparse
import chess
import torch

from nn_model import ChessEvaluatorNN, load_model
from search import find_best_move, _quiescence

TEST_POSITIONS = [
    # (namn, FEN, "bra drag" som räddar pjäsen, "dåligt drag" som hänger den)
    ("Dam hänger för springare", "4k3/6p1/5n2/3Q4/8/8/7P/4K3 w - - 0 1", "d5d1", "h2h3"),
    ("Torn hänger för löpare",   "4k3/8/8/8/3b4/8/3R3P/4K3 w - - 0 1", "d2d5", "h2h3"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="chess_nnue.pth")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessEvaluatorNN().to(device)
    if not load_model(model, args.model_path, device=device):
        print("Kunde inte ladda modellen -- avbryter.")
        return
    model.eval()

    for name, fen, good_uci, bad_uci in TEST_POSITIONS:
        board = chess.Board(fen)
        print(f"\n=== {name} ===")
        print(f"FEN: {fen}")

        for depth in [1, 2, 3]:
            move, score = find_best_move(board.copy(), model, depth=depth, device=device)
            san = board.san(move)
            verdict = "OK (räddar pjäsen)" if move.uci() == good_uci else (
                "*** HÄNGER PJÄSEN ***" if move.uci() == bad_uci else "(annat drag)")
            print(f"  depth={depth}: valt drag = {san:8s} score={score:+.2f}   {verdict}")

        # Visa exakt vad nätverket tycker om att göra-inget vs rädda pjäsen
        good_move = chess.Move.from_uci(good_uci)
        bad_move = chess.Move.from_uci(bad_uci)
        for label, mv in [("Rädda pjäsen", good_move), ("Ignorera hotet", bad_move)]:
            b2 = board.copy()
            b2.push(mv)
            val = _quiescence(b2, -float("inf"), float("inf"), model, device, -1)
            print(f"    {label:16s} ({b2.peek()}) -> nätverkets värde efter draget (vits persp) = {-val:+.2f}")


if __name__ == "__main__":
    main()
