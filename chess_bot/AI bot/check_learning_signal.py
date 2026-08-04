"""
check_learning_signal.py
--------------------------
Tar ett slumpat urval av RIKTIGA positioner ur din replay buffer och jämför:
  1. Modellens prediktion vs. det sanna Stockfish-facit-värdet (korrelation + MSE)
  2. En trivial baslinje som bara alltid gissar medelvärdet av alla targets

Om modellens MSE inte är betydligt bättre än baslinjens MSE, och korrelationen
är nära 0, har nätverket i praktiken inte lärt sig något -- oavsett hur mycket
vikterna har rört sig.

    python check_learning_signal.py --model-path chess_nnue.pth --buffer-path replay_buffer.pt
"""

import argparse
import torch

from nn_model import ChessEvaluatorNN, load_model
from replay_buffer import load_buffer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="chess_nnue.pth")
    p.add_argument("--buffer-path", type=str, default="replay_buffer.pt")
    p.add_argument("--n-samples", type=int, default=5000)
    args = p.parse_args()

    device = "cpu"
    model = ChessEvaluatorNN().to(device)
    load_model(model, args.model_path, device=device)
    model.eval()

    X, y = load_buffer(args.buffer_path)
    print(f"Buffer: {X.shape[0]} positioner totalt.\n")

    n = min(args.n_samples, X.shape[0])
    idx = torch.randperm(X.shape[0])[:n]
    Xs, ys = X[idx], y[idx]

    with torch.no_grad():
        preds = model(Xs).squeeze(1)

    # Modellens fel
    model_mse = ((preds - ys) ** 2).mean().item()

    # Trivial baslinje: gissa alltid medelvärdet
    baseline_pred = ys.mean()
    baseline_mse = ((baseline_pred - ys) ** 2).mean().item()

    # Korrelation mellan prediktion och facit
    pred_c = preds - preds.mean()
    y_c = ys - ys.mean()
    corr = (pred_c * y_c).sum() / (pred_c.norm() * y_c.norm() + 1e-8)

    print(f"Antal testade positioner: {n}")
    print(f"Sant facit -- medel: {ys.mean():.3f}, std: {ys.std():.3f}, min: {ys.min():.3f}, max: {ys.max():.3f}")
    print(f"Modellens prediktion -- medel: {preds.mean():.3f}, std: {preds.std():.3f}, "
          f"min: {preds.min():.3f}, max: {preds.max():.3f}")
    print()
    print(f"Modellens MSE:        {model_mse:.4f}")
    print(f"Baslinje-MSE (medel): {baseline_mse:.4f}")
    print(f"Korrelation (pred vs sant facit): {corr.item():.4f}")
    print()
    if model_mse >= baseline_mse * 0.95:
        print(">>> Modellen är INTE bättre än att bara gissa medelvärdet. Den har inte lärt sig något användbart.")
    elif corr.item() < 0.2:
        print(">>> Svag/obefintlig korrelation -- modellen har knappt lärt sig något, trots lägre MSE än baslinjen.")
    else:
        print(">>> Modellen visar verklig inlärningssignal på träningsdata. Problemet ligger då troligen i "
              "generalisering till ovanliga positioner, eller i sökningen/quiescence, inte i grundträningen.")


if __name__ == "__main__":
    main()