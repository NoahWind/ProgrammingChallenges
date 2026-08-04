"""
replay_buffer.py
-----------------
Enkel disk-backad "Experience Replay Buffer": (position_tensor, stockfish_eval)-par
som samlas in över flera träningskörningar, sparas mellan sessioner, och begränsas
till ett max-antal så att filen inte växer obegränsat.
"""

from __future__ import annotations
import os
import torch

from nn_model import INPUT_SIZE


def load_buffer(path: str):
    """Returnerar (X, y) som tensorer. Tomma tensorer om filen inte finns."""
    if os.path.exists(path):
        data = torch.load(path)
        return data["X"], data["y"]
    return torch.empty((0, INPUT_SIZE), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)


def save_buffer(path: str, X: torch.Tensor, y: torch.Tensor) -> None:
    torch.save({"X": X, "y": y}, path)


def append_and_cap(
    X: torch.Tensor,
    y: torch.Tensor,
    new_positions: list[tuple[torch.Tensor, float]],
    max_size: int = 300_000,
) -> tuple[torch.Tensor, float]:
    """Lägger till nya (tensor, target)-par i bufferten och klipper till max_size.

    Om bufferten överstiger max_size slumpas ett urval ut (behåller mångfald
    istället för att bara droppa de äldsta -- viktigt eftersom "äldst" ofta
    motsvarar tidiga, svagare partier).
    """
    if new_positions:
        new_X = torch.stack([p[0] for p in new_positions])
        new_y = torch.tensor([p[1] for p in new_positions], dtype=torch.float32)
        X = torch.cat([X, new_X], dim=0)
        y = torch.cat([y, new_y], dim=0)

    if X.shape[0] > max_size:
        keep_idx = torch.randperm(X.shape[0])[:max_size]
        X = X[keep_idx]
        y = y[keep_idx]

    return X, y
