"""
train.py
--------
Tränar ChessEvaluatorNN genom "Knowledge Distillation" från Stockfish.
"""

from __future__ import annotations
import argparse
import random
import time

import chess
import chess.engine
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nn_model import ChessEvaluatorNN, board_to_tensor, save_model, load_model, stockfish_score_to_pawns
from search import find_best_move
from replay_buffer import load_buffer, save_buffer, append_and_cap


def parse_args():
    p = argparse.ArgumentParser(description="Träna schack-NN mot Stockfish (knowledge distillation).")
    p.add_argument("--stockfish-path", type=str, default="stockfish-windows-x86-64-avx2.exe",
                    help="Sökväg till Stockfish-binären.")
    p.add_argument("--model-path", type=str, default="chess_nnue.pth")
    p.add_argument("--buffer-path", type=str, default="replay_buffer.pt")

    p.add_argument("--games", type=int, default=4000, help="Antal partier att spela totalt i denna körning.")
    p.add_argument("--max-moves", type=int, default=120, help="Max halvdrag per parti innan det avbryts.")
    p.add_argument("--opening-random-moves", type=int, default=4,
                    help="Antal slumpade öppningsdrag för variation (0-N).")

    p.add_argument("--game-depth", type=int, default=3, help="Sökdjup för botten (negamax, plies).")
    p.add_argument("--stockfish-move-depth", type=int, default=8,
                    help="Djup Stockfish söker på när den VÄLJER drag (motståndaren).")
    p.add_argument("--stockfish-eval-depth", type=int, default=12,
                    help="Djup Stockfish söker på när den GER FACIT för träningsdata (bör vara djupare).")
    p.add_argument("--skill-level", type=int, default=10, help="Stockfish Skill Level 0-20 (motståndarens styrka).")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Mapp där alla checkpoints sparas.")

    p.add_argument("--train-every", type=int, default=50, help="Träna nätverket var N:e parti.")
    p.add_argument("--checkpoint-every", type=int, default=200, help="Spara en namngiven checkpoint var N:e parti.")
    p.add_argument("--epochs", type=int, default=3, help="Antal epoker per träningspass.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4, help="Inlärningshastighet (sänkt till 1e-4 för stabilitet).")
    p.add_argument("--max-buffer-size", type=int, default=300_000)

    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def play_training_game(board_depth_bot: int, engine, args, model, device) -> tuple[list, str]:
    """Spelar ETT parti: botten mot Stockfish. Returnerar (positioner, resultat)."""
    board = chess.Board()
    bot_color = random.choice([chess.WHITE, chess.BLACK])

    for _ in range(random.randint(0, args.opening_random_moves)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))

    positions: list[tuple[torch.Tensor, float]] = []
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < args.max_moves:
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=args.stockfish_eval_depth))
            target = stockfish_score_to_pawns(info["score"])
            positions.append((board_to_tensor(board), target))
        except chess.engine.EngineTerminatedError:
            break

        if board.turn == bot_color:
            move, _ = find_best_move(board, model, depth=board_depth_bot, device=device)
        else:
            result = engine.play(board, chess.engine.Limit(depth=args.stockfish_move_depth))
            move = result.move

        if move is None:
            break
        board.push(move)
        move_count += 1

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    return positions, result


def train_on_buffer(model, X, y, optimizer, loss_fn, args, device):
    if X.shape[0] < args.batch_size:
        print(f"[train] Bufferten har bara {X.shape[0]} positioner -- hoppar över träningssteget "
              f"(behöver minst batch-size={args.batch_size}).")
        return

    n = X.shape[0]
    n_val = max(1, int(0.1 * n))
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            # Gradient clipping för att förhindra enorma stöt-steg vid outliers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val.unsqueeze(1)).item()
        model.train()

        print(f"[train]   epoch {epoch + 1}/{args.epochs}  "
              f"train_loss={total_loss / max(1, n_batches):.4f}  val_loss={val_loss:.4f}")

    model.eval()


def main():
    import os
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Använder device: {device}")

    model = ChessEvaluatorNN().to(device)
    load_model(model, args.model_path, device=device)
    model.eval()

    # Skapa Optimizer och Loss-funktion EN gång här i main()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    X, y = load_buffer(args.buffer_path)
    print(f"[train] Laddade replay buffer med {X.shape[0]} sparade positioner.")

    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    except FileNotFoundError:
        print(f"[train] FEL: Hittade inte Stockfish på '{args.stockfish_path}'. "
              f"Ange rätt sökväg med --stockfish-path.")
        return
    engine.configure({"Skill Level": max(0, min(20, args.skill_level))})

    games_since_train = 0

    try:
        for game_idx in range(1, args.games + 1):
            t0 = time.time()
            positions, result = play_training_game(args.game_depth, engine, args, model, device)
            dt = time.time() - t0

            X, y = append_and_cap(X, y, positions, max_size=args.max_buffer_size)
            games_since_train += 1

            print(f"[train] Parti {game_idx}/{args.games}  resultat={result}  "
                  f"drag={len(positions)}  tid={dt:.1f}s  buffer={X.shape[0]}")

            if games_since_train >= args.train_every:
                start = time.time()
                print(f"[train] -- Tränar nätverket på {X.shape[0]} positioner --")
                
                # Här skickas optimizer och loss_fn in till funktionen
                train_on_buffer(model, X, y, optimizer, loss_fn, args, device)
                
                save_model(model, args.model_path)
                save_buffer(args.buffer_path, X, y)
                games_since_train = 0
                end = time.time()
                print(f"[train] -- Träning klar, sparade modell och buffer. Tid: {end - start:.1f}s --")

            if game_idx % args.checkpoint_every == 0:
                time.sleep(20)
                base_name = os.path.basename(args.model_path).replace(".pth", f"_ep_{game_idx}.pth")
                ckpt_path = os.path.join(args.checkpoint_dir, base_name)
                save_model(model, ckpt_path)
                print(f"[train] Sparade checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print("\n[train] Avbrutet av användaren -- sparar innan avslut...")
    finally:
        engine.quit()
        save_model(model, args.model_path)
        save_buffer(args.buffer_path, X, y)
        print(f"[train] Klart. Modell sparad till '{args.model_path}', "
              f"buffer sparad till '{args.buffer_path}' ({X.shape[0]} positioner totalt).")


if __name__ == "__main__":
    main()