"""
nn_model.py
-----------
Nätverksarkitektur (ChessEvaluatorNN), ställning -> tensor-kodning (board_to_tensor),
samt hjälpfunktioner för att spara/ladda vikter.

Konventionen genom hela projektet:
  * Nätverkets output är alltid en utvärdering i "pjäsvärdes-enheter" (pawns),
    ur VITS perspektiv (positivt = vitt står bättre), oavsett vem som har draget.
  * Det är search.py:s ansvar att konvertera detta till "sida-i-draget"-relativt
    värde (negamax-konvention) genom att multiplicera med +1/-1.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import chess

# ---------------------------------------------------------------------------
# Bräde -> tensor
# ---------------------------------------------------------------------------

_PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# 12 pjäsplan (6 pjästyper x 2 färger) x 8x8 rutor = 768
# + 7 extra features (se nedan) = 775
NUM_PIECE_PLANES = 12
EXTRA_FEATURES = 7
INPUT_SIZE = NUM_PIECE_PLANES * 64 + EXTRA_FEATURES


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Konverterar ett chess.Board till en platt float32-tensor av längd INPUT_SIZE.

    Plan 0-5:  vita bönder, springare, löpare, torn, dam, kung
    Plan 6-11: svarta bönder, springare, löpare, torn, dam, kung
    Extra features:
        0: vem har draget (1.0 = vit, 0.0 = svart)
        1: vit kan rockera kort
        2: vit kan rockera lång
        3: svart kan rockera kort
        4: svart kan rockera lång
        5: sida i draget står i schack
        6: draget nummer, normaliserat (fullmove_number / 100)
    """
    planes = np.zeros((NUM_PIECE_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane_idx = _PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane_idx += 6
        planes[plane_idx, row, col] = 1.0

    extra = np.zeros(EXTRA_FEATURES, dtype=np.float32)
    extra[0] = 1.0 if board.turn == chess.WHITE else 0.0
    extra[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    extra[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    extra[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    extra[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    extra[5] = 1.0 if board.is_check() else 0.0
    extra[6] = min(board.fullmove_number, 200) / 100.0

    flat_planes = torch.from_numpy(planes.reshape(-1))
    extra_t = torch.from_numpy(extra)
    return torch.cat([flat_planes, extra_t])


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------


class ChessEvaluatorNN(nn.Module):
    """Enkel MLP-utvärderare (NNUE-inspirerad, men utan HalfKP-features).

    Input:  (batch, INPUT_SIZE)
    Output: (batch, 1) -- utvärdering i pawns, ur vits perspektiv.
    """

    def __init__(self, input_size: int = INPUT_SIZE, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Spara / ladda
# ---------------------------------------------------------------------------


def save_model(model: nn.Module, path: str = "chess_nnue.pth") -> None:
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str = "chess_nnue.pth", device: str = "cpu") -> bool:
    """Laddar sparade vikter in i `model` om filen finns. Returnerar True/False."""
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[nn_model] Laddade sparade modellvikter från '{path}'")
        return True
    print(f"[nn_model] Ingen sparad modell hittades vid '{path}' -- startar med slumpade vikter.")
    return False


# ---------------------------------------------------------------------------
# Stockfish score -> träningsmål
# ---------------------------------------------------------------------------


def stockfish_score_to_pawns(score, clip: float = 10.0) -> float:
    """Konverterar ett chess.engine.PovScore till ett white-relativt pawns-värde.

    Matt-poäng klipps till `clip` (med rätt tecken) så att nätverket inte försöker
    lära sig skillnaden mellan "matt om 1" och "matt om 20" -- båda är bara "vinnande".
    """
    white_score = score.white()
    cp = white_score.score(mate_score=int(clip * 100) + 100)
    pawns = cp / 100.0
    return max(-clip, min(clip, pawns))