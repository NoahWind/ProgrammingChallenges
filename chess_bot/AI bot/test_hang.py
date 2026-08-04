import chess
import torch
import torch.nn as nn

from search import find_best_move, evaluate, _quiescence
from nn_model import board_to_tensor

PIECE_VALUES = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0]  # pawn,knight,bishop,rook,queen,king


class PerfectMaterialModel(nn.Module):
    """Ignores the network entirely -- computes exact material balance
    (white planes positive, black planes negative) straight from the
    board_to_tensor encoding. If search still hangs a piece with THIS
    'model', the bug is in search.py, not in NN quality."""

    def forward(self, x):
        batch = x.shape[0]
        planes = x[:, :768].view(batch, 12, 64)
        counts = planes.sum(dim=2)  # (batch, 12)
        vals = torch.tensor(PIECE_VALUES + PIECE_VALUES)
        white = (counts[:, :6] * vals[:6]).sum(dim=1)
        black = (counts[:, 6:] * vals[6:]).sum(dim=1)
        return (white - black).unsqueeze(1)


model = PerfectMaterialModel()
model.eval()
device = "cpu"

# White queen on d5, undefended, attacked by black knight on f6.
# Knight f6 IS defended by pawn g7 (so simply grabbing the knight is a bad trade).
# White to move. h2h3 ignores the threat and hangs the queen for free.
# Qd5-d1 (or similar) saves it. A depth-1 search with correct quiescence
# should clearly prefer saving the queen.
fen = "4k3/6p1/5n2/3Q4/8/8/7P/4K3 w - - 0 1"
board = chess.Board(fen)

print("Position:")
print(board)
print()
print("Static eval (white persp):", evaluate(board, model, device))
print()

for depth in [1, 2, 3]:
    move, score = find_best_move(board.copy(), model, depth=depth, device=device)
    san = board.san(move)
    print(f"depth={depth}: chosen move = {san:8s} score(side-to-move persp) = {score:+.2f}"
          f"  {'<<< HUNG THE QUEEN' if san in ('h3','h4') else ''}")

print()
print("--- Sanity: manually check candidate moves at depth=1 ---")
for move in board.legal_moves:
    san = board.san(move)
    if san not in ("h3", "h4", "Qd1", "Qd2", "Qd3", "Qd4", "Qxf6+", "Qa5+", "Qb5+", "Qc5+",
                    "Qe5+", "Qxd8+", "Qd6", "Qd7+", "Qc4", "Qb3", "Qa2", "Qe4", "Qf3", "Qg2", "Qh1"):
        continue
    b2 = board.copy()
    b2.push(move)
    perspective = -1  # black to move now, from white's original perspective it's flipped
    val = _quiescence(b2, -float("inf"), float("inf"), model, device, perspective)
    print(f"  {san:8s} -> negamax value after this move (black's persp) = {val:+.2f}"
          f"  (white persp = {-val:+.2f})")
