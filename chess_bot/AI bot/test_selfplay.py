"""
test_selfplay.py
-----------------
Låter en (eller två) tränade modeller spela mot sig själva/varandra, för att
visuellt och kvantitativt utvärdera hur bra nätverket blivit sedan förra
träningsrundan.

Loggar varje halvdrag till en CSV-fil (FEN, utvärdering, valt drag) och sparar
alla partier som PGN så du kan öppna dem i Arena/lichess/ChessBase.

Körexempel:
    python test_selfplay.py --model-path chess_nnue.pth --games 5 --depth 2

Om du vill jämföra två olika checkpoints mot varandra:
    python test_selfplay.py --model-path chess_nnue.pth --model-b-path chess_nnue_ep_25.pth
"""

from __future__ import annotations
import argparse
import csv
import os
import random
import time

import chess
import chess.pgn
import torch

from nn_model import ChessEvaluatorNN, load_model
from search import find_best_move, position_key


def parse_args():
    p = argparse.ArgumentParser(description="Självspel + FEN-loggning för utvärdering av tränat nätverk.")
    p.add_argument("--model-path", type=str, default="chess_nnue.pth", help="Modell för VIT.")
    p.add_argument("--model-b-path", type=str, default=None,
                    help="Valfri separat modell för SVART (för att jämföra två checkpoints). "
                         "Om ej satt används samma modell för båda sidor.")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--depth", type=int, default=2, help="Sökdjup i negamax.")
    p.add_argument("--max-moves", type=int, default=150)
    p.add_argument("--opening-random-moves", type=int, default=1)
    p.add_argument("--csv-path", type=str, default="test_fen_log.csv")
    p.add_argument("--pgn-path", type=str, default="self_play_tests.pgn")
    p.add_argument("--print-board", action="store_true", default=True)
    p.add_argument("--no-print-board", dest="print_board", action="store_false")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def play_one_game(game_idx, model_white, model_black, args, device, csv_writer, pgn_file):
    board = chess.Board()

    for _ in range(random.randint(0, args.opening_random_moves)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))

    history = [position_key(board)]
    move_number = 0
    while not board.is_game_over(claim_draw=True) and move_number < args.max_moves:
        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        model = model_white if board.turn == chess.WHITE else model_black

        move, score = find_best_move(board, model, depth=args.depth, device=device, history=history)
        if move is None:
            break

        fen_before = board.fen()
        san = board.san(move)
        board.push(move)
        history.append(position_key(board))
        move_number += 1

        csv_writer.writerow([
            game_idx, move_number, side_to_move, fen_before, f"{score:.3f}", san, move.uci(),
        ])

        if args.print_board:
            print(f"\n--- Parti {game_idx}, drag {move_number} ({side_to_move}): {san}  "
                  f"(eval={score:+.2f}) ---")
            print(board)

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"

    game = chess.pgn.Game()
    game.headers["Event"] = "NN Self-Play Test"
    game.headers["White"] = "NN Bot (model A)"
    game.headers["Black"] = "NN Bot (model B)" if model_black is not model_white else "NN Bot (model A)"
    game.headers["Result"] = result
    node = game
    for mv in board.move_stack:
        node = node.add_variation(mv)
    print(game, file=pgn_file, end="\n\n")

    return result


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Använder device: {device}")

    model_white = ChessEvaluatorNN().to(device)
    load_model(model_white, args.model_path, device=device)
    model_white.eval()

    if args.model_b_path:
        model_black = ChessEvaluatorNN().to(device)
        load_model(model_black, args.model_b_path, device=device)
        model_black.eval()
    else:
        model_black = model_white

    csv_is_new = not os.path.exists(args.csv_path)
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}

    with open(args.csv_path, "a", newline="", encoding="utf-8") as csv_f, \
         open(args.pgn_path, "a", encoding="utf-8") as pgn_f:

        writer = csv.writer(csv_f)
        if csv_is_new:
            writer.writerow(["game", "move_number", "side_to_move", "fen", "eval", "move_san", "move_uci"])

        for game_idx in range(1, args.games + 1):
            t0 = time.time()
            result = play_one_game(game_idx, model_white, model_black, args, device, writer, pgn_f)
            results[result] = results.get(result, 0) + 1
            print(f"\n[test] Parti {game_idx}/{args.games} klart. Resultat: {result}  "
                  f"(tid: {time.time() - t0:.1f}s)")

    print("\n[test] Sammanfattning:")
    print(f"  Vit vann : {results.get('1-0', 0)}")
    print(f"  Svart vann: {results.get('0-1', 0)}")
    print(f"  Remi      : {results.get('1/2-1/2', 0)}")
    print(f"  Avbrutna  : {results.get('*', 0)}")
    print(f"\n[test] FEN-logg sparad till '{args.csv_path}'")
    print(f"[test] PGN sparad till '{args.pgn_path}'")


if __name__ == "__main__":
    main()