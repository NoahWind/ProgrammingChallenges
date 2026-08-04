"""
search.py
---------
Negamax med alpha-beta-beskärning och en enkel quiescence-sökning för fångster,
där bladnoder utvärderas av det neurala nätverket (nn_model.ChessEvaluatorNN).

Detta är motorn som faktiskt VÄLJER drag -- både när botten spelar mot Stockfish
under träning, och när den spelar mot sig själv i test_selfplay.py.

Prestanda-notis: varje evaluate()-anrop är en forward-pass genom nätverket, vilket
är mycket dyrare än en handskriven utvärderingsfunktion. Håll game_depth lågt
(1-3) tills du har anledning att optimera (batchad evaluering, mindre nät, etc).
"""

from __future__ import annotations
import chess
import torch

from nn_model import board_to_tensor

MATE_VALUE = 100000.0


def evaluate(board: chess.Board, model, device: str = "cpu") -> float:
    """Utvärdering ur VITS perspektiv (positivt = bra för vit)."""
    if board.is_checkmate():
        # Sida i draget är mattad -> mycket dåligt för den sidan.
        return -MATE_VALUE if board.turn == chess.WHITE else MATE_VALUE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    with torch.no_grad():
        t = board_to_tensor(board).unsqueeze(0).to(device)
        value = model(t).item()
    return value


def _order_moves(board: chess.Board):
    """Enkel MVV-LVA-ordning: fångster med hög offervärde/lågt anfallarvärde först."""

    def score_move(move: chess.Move) -> int:
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v = victim.piece_type if victim else 0
            a = attacker.piece_type if attacker else 0
            return 10 * v - a
        if move.promotion:
            return 5
        return 0

    return sorted(board.legal_moves, key=score_move, reverse=True)


def _quiescence(
    board: chess.Board,
    alpha: float,
    beta: float,
    model,
    device: str,
    perspective: int,
    depth: int = 0,
    max_depth: int = 4,
) -> float:
    stand_pat = perspective * evaluate(board, model, device)
    if depth >= max_depth:
        return stand_pat
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, model, device, -perspective, depth + 1, max_depth)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    model,
    device: str,
    perspective: int,
) -> float:
    if board.is_checkmate():
        return -MATE_VALUE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth == 0:
        return _quiescence(board, alpha, beta, model, device, perspective)

    best = -float("inf")
    for move in _order_moves(board):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, model, device, -perspective)
        board.pop()
        if score > best:
            best = score
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def find_best_move(board: chess.Board, model, depth: int = 2, device: str = "cpu"):
    """Returnerar (best_move, score) där score är sida-i-draget-relativt (negamax-konvention).

    Returnerar (None, 0.0) om det inte finns några lagliga drag.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0

    perspective = 1 if board.turn == chess.WHITE else -1
    alpha, beta = -float("inf"), float("inf")
    best_move = None
    best_score = -float("inf")

    for move in _order_moves(board):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, model, device, -perspective)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    if best_move is None:
        best_move = legal_moves[0]
    return best_move, best_score
