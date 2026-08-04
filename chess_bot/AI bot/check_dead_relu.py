"""
check_dead_relu.py
-------------------
Kör ett gäng slumpade/riktiga ställningar genom nätverket och räknar hur stor
andel av varje ReLU-lagers neuroner som ALDRIG aktiveras (dvs. är "döda").

    python check_dead_relu.py --model-path chess_nnue.pth
"""

import argparse
import random
import chess
import torch

from nn_model import ChessEvaluatorNN, load_model, board_to_tensor


def random_legal_position(max_moves=30):
    board = chess.Board()
    for _ in range(random.randint(0, max_moves)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="chess_nnue.pth")
    p.add_argument("--n-samples", type=int, default=200)
    args = p.parse_args()

    device = "cpu"
    model = ChessEvaluatorNN().to(device)
    load_model(model, args.model_path, device=device)
    model.eval()

    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            act = (out > 0).float()
            if name not in activations:
                activations[name] = act.sum(dim=0)
            else:
                activations[name] += act.sum(dim=0)
        return hook

    relu_layers = [m for m in model.net if isinstance(m, torch.nn.ReLU)]
    for i, layer in enumerate(relu_layers):
        layer.register_forward_hook(make_hook(f"relu_{i}"))

    boards = [random_legal_position() for _ in range(args.n_samples)]
    tensors = torch.stack([board_to_tensor(b) for b in boards])

    with torch.no_grad():
        model(tensors)

    print(f"Testade {args.n_samples} slumpade ställningar.\n")
    for name, counts in activations.items():
        total = counts.numel()
        dead = (counts == 0).sum().item()
        print(f"{name}: {dead}/{total} neuroner AKTIVERADES ALDRIG "
              f"({100 * dead / total:.1f}% döda)")


if __name__ == "__main__":
    main()