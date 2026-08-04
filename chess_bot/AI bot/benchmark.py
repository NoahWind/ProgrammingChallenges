"""
benchmark.py
------------
Objektivt mått på framsteg: spelar N partier mellan nätverket och Stockfish på
en FAST Skill Level, med alternerande färger, och rapporterar vinst/remi/förlust.

Kör detta med SAMMA --skill-level varje gång (t.ex. efter var 5:e träningssession)
för att kunna jämföra resultat rakt av mellan körningar -- det är så du vet om
nätverket faktiskt blir bättre, inte bara att träningsförlusten sjunker.

Körexempel:
    python benchmark.py --model-path chess_nnue.pth --games 20 --skill-level 5
"""

from __future__ import annotations
import argparse
import random
import time

import chess
import chess.engine
import torch

from nn_model import ChessEvaluatorNN, load_model
from search import find_best_move


def parse_args():
    p = argparse.ArgumentParser(description="Benchmarka nätverket mot Stockfish på fast styrka.")
    p.add_argument("--stockfish-path", type=str, default="stockfish-windows-x86-64-avx2.exe")
    p.add_argument("--model-path", type=str, default="safes/chess_nnue.pth")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--depth", type=int, default=1, help="Sökdjup för botten.")
    p.add_argument("--stockfish-move-depth", type=int, default=6)
    p.add_argument("--skill-level", type=int, default=5)
    p.add_argument("--max-moves", type=int, default=150)
    return p.parse_args()


def play_benchmark_game(bot_color, engine, model, args, device):
    board = chess.Board()
    move_count = 0
    while not board.is_game_over(claim_draw=True) and move_count < args.max_moves:
        if board.turn == bot_color:
            move, _ = find_best_move(board, model, depth=args.depth, device=device)
        else:
            result = engine.play(board, chess.engine.Limit(depth=args.stockfish_move_depth))
            move = result.move
        if move is None:
            break
        board.push(move)
        move_count += 1
    return board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ChessEvaluatorNN().to(device)
    if not load_model(model, args.model_path, device=device):
        print("[benchmark] Ingen tränad modell hittades -- benchmarkar en slumpad modell (endast som referens).")
    model.eval()

    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    except FileNotFoundError:
        print(f"[benchmark] FEL: Hittade inte Stockfish på '{args.stockfish_path}'.")
        return
    engine.configure({"Skill Level": max(0, min(20, args.skill_level))})

    wins = draws = losses = 0
    try:
        for i in range(1, args.games + 1):
            bot_color = chess.WHITE if i % 2 == 1 else chess.BLACK
            t0 = time.time()
            result = play_benchmark_game(bot_color, engine, model, args, device)

            if result == "1/2-1/2":
                draws += 1
                outcome = "REMI"
            elif (result == "1-0" and bot_color == chess.WHITE) or (result == "0-1" and bot_color == chess.BLACK):
                wins += 1
                outcome = "VINST"
            elif result == "*":
                outcome = "AVBRUTET"
            else:
                losses += 1
                outcome = "FÖRLUST"

            color_str = "vit" if bot_color == chess.WHITE else "svart"
            print(f"[benchmark] Parti {i}/{args.games} (botten spelade {color_str}): "
                  f"{outcome}  ({time.time() - t0:.1f}s)")
    finally:
        engine.quit()

    played = wins + draws + losses
    score = wins + 0.5 * draws
    print("\n[benchmark] ===== RESULTAT =====")
    print(f"  Skill Level: {args.skill_level}   Sökdjup: {args.depth}")
    print(f"  Vinster: {wins}  Remi: {draws}  Förluster: {losses}  (av {played} avslutade partier)")
    if played > 0:
        print(f"  Poäng: {score}/{played} = {100 * score / played:.1f}%")


if __name__ == "__main__":
    main()
