"""
search.py
---------
Produktionsklar schacksökning för en motor där lövnoder evalueras av ett
neuralt nätverk (nn_model.ChessEvaluatorNN) i stället för en handskriven
statisk evalueringsfunktion.

Arkitektur:
    * Negamax med alpha-beta-beskärning.
    * Principal Variation Search (PVS): första draget söks med fullt fönster,
      resterande med nollfönster + re-search vid behov.
    * Late Move Reductions (LMR) med djup-/index-beroende reduktion samt
      Late Move Pruning (LMP) av sena tysta drag på grunda noder.
    * Quiescence Search med strikt MVV-LVA-ordning, delta pruning och
      gallring av förlorande slag — minimerar antalet NN-anrop.
    * Null Move Pruning (klassisk, evalueringsfri variant — kräver INGET
      extra nätverksanrop, till skillnad från static null move / reverse
      futility pruning som medvetet är uteslutna).
    * Transpositionstabell (TT) med Zobrist-hash, EXACT/LOWER/UPPER-flaggor,
      ply-justerade mattpoäng och FIFO-utrensning (aldrig .clear() mitt i
      en sökning).
    * History-heuristik för ordning av tysta drag + killer moves.
    * Iterativ fördjupning i roten med batch-evaluering av rotdragen för en
      stark initial dragordning — utan tunga board.copy()-anrop.

Prestandaregler som följs strikt:
    * Inga board.copy() i rekursiva grenar — endast push()/pop().
    * Zobrist-nyckeln beräknas EN gång per nod och skickas ned till barnet
      (halverar antalet hash-beräkningar).
    * gives_check() anropas lat — endast när LMR/LMP-villkoren i övrigt
      är uppfyllda (det är ett dyrt anrop).
    * Matt/patt detekteras via dragloopen (inga separata is_checkmate/
      is_stalemate-anrop per nod).
    * Platta hjälpfunktioner (ej nästlade) för minimal Python-overhead.
"""

from __future__ import annotations

import time

import chess
import chess.polyglot
import torch

from nn_model import board_to_tensor

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

INF = float("inf")

# Mattpoäng. MATE_THRESHOLD skiljer "riktiga" mattpoäng från vanliga
# evalueringar så att TT kan ply-justera dem korrekt.
MATE_VALUE = 100_000.0
MATE_THRESHOLD = MATE_VALUE - 1_000.0  # allt över detta räknas som matt-nära

# Straff (i pawns) för att gå in i en redan besökt ställning (dragupprepning).
REPETITION_PENALTY = 1.5

# --- Null move pruning -----------------------------------------------------
NULL_MOVE_MIN_DEPTH = 3    # kräv minst detta restdjup
NULL_MOVE_REDUCTION = 2    # R: sök nollfönstret med depth - 1 - R

# --- Quiescence ------------------------------------------------------------
QSEARCH_MAX_PLY = 4        # maxdjup i quiescence
DELTA_MARGIN = 2.0         # delta pruning-marginal (i pawns)

# Enkla pjäsvärden (pawns) för delta pruning och gallring av förlorande slag —
# endast heuristisk gallring, aldrig slutlig evaluering (den sköts av nätverket).
_PIECE_VALUES = (0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 0.0)  # index = chess.PieceType

# --- Late Move Reductions / Pruning ------------------------------------------
LMR_MIN_DEPTH = 3          # reducera bara om restdjupet är minst detta
LMR_MOVE_THRESHOLD = 3     # reducera först från och med detta dragindex

LMP_MAX_DEPTH = 2          # LMP endast på grunda noder
LMP_BASE = 7               # hoppa över tysta drag efter LMP_BASE + 4*depth st

# --- Iterativ fördjupning ----------------------------------------------------
ITER_TIME_GROWTH_ESTIMATE = 4.0  # gissad tillväxtfaktor per extra djup

# --- Rotgallring ---------------------------------------------------------------
# Drag som i föregående iteration låg mer än marginalen under bästa draget
# söks inte på djupare nivåer — vi vet redan att de är dåliga. Marginalen
# krymper med djupet (djupare sökning = pålitligare poäng = hårdare gallring).
ROOT_PRUNE_MIN_DEPTH = 3     # börja gallra först från denna iteration
ROOT_PRUNE_MARGIN_BASE = 2.5 # marginal (pawns) vid första gallringsdjupet
ROOT_PRUNE_MARGIN_STEP = 0.5 # minskning av marginalen per extra djup
ROOT_PRUNE_MARGIN_MIN = 1.0  # golvet för marginalen
ROOT_MIN_MOVES = 5           # sök alltid minst så här många kandidater

MAX_PLY = 128  # maximalt sökdjup (dimensionerar killer-tabellen)

# ---------------------------------------------------------------------------
# Transpositionstabell & evalueringscache (FIFO-utrensning)
# ---------------------------------------------------------------------------

TT_EXACT = 0   # exakt poäng
TT_LOWER = 1   # fail-high: poängen är en undre gräns (score >= beta)
TT_UPPER = 2   # fail-low: poängen är en övre gräns (score <= alpha)

# key -> (depth, score, flag, best_move)
_TT: dict[int, tuple[int, float, int, chess.Move | None]] = {}
MAX_TT_ENTRIES = 300_000

# key -> rå NN-utvärdering (ur vits perspektiv)
_EVAL_CACHE: dict[int, float] = {}
MAX_EVAL_CACHE_ENTRIES = 300_000

# History-heuristik: [color][from*64+to] -> ackumulerad cutoff-vikt.
# Förallokerade platta listor — inga dict-uppslag i heta loopar.
_HISTORY: list[list[int]] = [[0] * 4096, [0] * 4096]
_HISTORY_MAX = 1 << 20  # skala ned vid overflow så ordningen förblir stabil


def clear_search_caches() -> None:
    """Töm TT, evalueringscache och history. Anropas ALDRIG mitt i en
    sökning — endast mellan partier eller när modellen byts ut."""
    _TT.clear()
    _EVAL_CACHE.clear()
    _HISTORY[0] = [0] * 4096
    _HISTORY[1] = [0] * 4096


def position_key(board: chess.Board) -> int:
    """Zobrist-hash (64-bitars heltal) via python-chess inbyggda polyglot-hash."""
    return chess.polyglot.zobrist_hash(board)


def _tt_store(key: int, depth: int, score: float, flag: int,
              move: chess.Move | None, ply: int) -> None:
    """Lagra en TT-post. Mattpoäng normaliseras till 'avstånd från roten av
    denna subsökning' så att de kan återjusteras korrekt vid uppslag från
    en annan ply. FIFO-utrensning i stället för .clear()."""
    if score >= MATE_THRESHOLD:
        score += ply
    elif score <= -MATE_THRESHOLD:
        score -= ply
    if len(_TT) >= MAX_TT_ENTRIES:
        _TT.pop(next(iter(_TT)))  # FIFO: släng äldsta posten, rensa aldrig allt
    _TT[key] = (depth, score, flag, move)


def _tt_score_from(entry_score: float, ply: int) -> float:
    """Återjustera en lagrad (rot-relativ) mattpoäng till aktuell ply."""
    if entry_score >= MATE_THRESHOLD:
        return entry_score - ply
    if entry_score <= -MATE_THRESHOLD:
        return entry_score + ply
    return entry_score


def _evict_eval_cache() -> None:
    if len(_EVAL_CACHE) >= MAX_EVAL_CACHE_ENTRIES:
        _EVAL_CACHE.pop(next(iter(_EVAL_CACHE)))


# ---------------------------------------------------------------------------
# Neural nätverks-evaluering (med cache och batchning)
# ---------------------------------------------------------------------------

def evaluate(board: chess.Board, model, device: str = "cpu") -> float:
    """Evaluera en ställning ur VITS perspektiv (publikt API).

    Terminala ställningar hanteras exakt (matt/patt/otillräckligt material);
    övriga skickas till nätverket, med FIFO-cache på Zobrist-nyckeln."""
    if board.is_checkmate():
        # Sidan vid draget är schackmatt.
        return -MATE_VALUE if board.turn == chess.WHITE else MATE_VALUE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    return _evaluate_nn(board, position_key(board), model, device)


def _evaluate_nn(board: chess.Board, key: int, model, device: str) -> float:
    """Rå NN-evaluering (ur vits perspektiv) med cache — inga terminalkontroller.
    Intern snabbväg för sökningen, där nyckeln redan är känd och
    terminala noder redan hanterats av dragloopen."""
    cached = _EVAL_CACHE.get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        t = board_to_tensor(board).unsqueeze(0).to(device)
        value = model(t).item()
    _evict_eval_cache()
    _EVAL_CACHE[key] = value
    return value


def evaluate_batch(boards: list[chess.Board], model, device: str = "cpu") -> list[float]:
    """Batch-evaluering av flera ställningar (ur vits perspektiv) i EN
    forward-pass. Terminala ställningar och cacheträffar hoppas över."""
    results = [0.0] * len(boards)
    to_eval_idx: list[int] = []
    to_eval_tensors: list[torch.Tensor] = []
    to_eval_keys: list[int] = []

    for i, b in enumerate(boards):
        if b.is_checkmate():
            results[i] = -MATE_VALUE if b.turn == chess.WHITE else MATE_VALUE
            continue
        if b.is_stalemate() or b.is_insufficient_material():
            results[i] = 0.0
            continue
        key = position_key(b)
        cached = _EVAL_CACHE.get(key)
        if cached is not None:
            results[i] = cached
            continue
        to_eval_idx.append(i)
        to_eval_keys.append(key)
        to_eval_tensors.append(board_to_tensor(b))

    if to_eval_tensors:
        with torch.no_grad():
            batch = torch.stack(to_eval_tensors).to(device)
            values = model(batch).squeeze(-1).tolist()
        if isinstance(values, float):  # squeeze av batchstorlek 1 ger en skalär
            values = [values]
        for idx, key, v in zip(to_eval_idx, to_eval_keys, values):
            results[idx] = v
            _evict_eval_cache()
            _EVAL_CACHE[key] = v

    return results


# ---------------------------------------------------------------------------
# Dragordning (platta hjälpfunktioner — inga nästlade closures i heta loopar)
# ---------------------------------------------------------------------------

def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA-poäng: värdefullaste offer först, billigaste angripare först."""
    if board.is_en_passant(move):
        return 10 * chess.PAWN - chess.PAWN
    victim = board.piece_type_at(move.to_square) or 0
    attacker = board.piece_type_at(move.from_square) or 0
    return 10 * victim - attacker


def _victim_value(board: chess.Board, move: chess.Move) -> float:
    """Materialvärde (pawns) på det slagna offret — för delta pruning."""
    if board.is_en_passant(move):
        return 1.0
    victim = board.piece_type_at(move.to_square)
    return _PIECE_VALUES[victim] if victim else 0.0


def _is_losing_capture(board: chess.Board, move: chess.Move) -> bool:
    """Grov SEE-approximation: slaget ser förlorande ut om angriparen är
    värdefullare än offret OCH rutan försvaras av motståndaren. Används
    endast för att gallra i quiescence — aldrig i huvudsökningen."""
    if board.is_en_passant(move):
        return False
    victim = board.piece_type_at(move.to_square) or 0
    attacker = board.piece_type_at(move.from_square) or 0
    if _PIECE_VALUES[attacker] <= _PIECE_VALUES[victim]:
        return False
    return board.is_attacked_by(not board.turn, move.to_square)


def _ordered_captures(board: chess.Board) -> list[chess.Move]:
    """Alla legala slag, sorterade strikt enligt MVV-LVA (för quiescence)."""
    captures = list(board.generate_legal_captures())
    captures.sort(key=lambda m: _mvv_lva(board, m), reverse=True)
    return captures


def _ordered_moves(
    board: chess.Board,
    tt_move: chess.Move | None,
    killer0: chess.Move | None,
    killer1: chess.Move | None,
) -> list[chess.Move]:
    """Fullständig dragordning för huvudsökningen:
    TT-drag > slag (MVV-LVA) > promoveringar > killers > history > övriga."""
    history = _HISTORY[board.turn]
    scored: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        if move == tt_move:
            score = 10_000_000
        elif board.is_capture(move):
            score = 1_000_000 + _mvv_lva(board, move)
        elif move.promotion is not None:
            score = 500_000
        elif move == killer0:
            score = 100_000
        elif move == killer1:
            score = 90_000
        else:
            score = history[(move.from_square << 6) | move.to_square]
        scored.append((score, move))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored]


def _store_killer(killers: list[list[chess.Move | None]], ply: int,
                  move: chess.Move) -> None:
    """Lagra ett killer move (två slots per ply, nyaste först)."""
    if ply < MAX_PLY:
        slot = killers[ply]
        if move != slot[0]:
            slot[1] = slot[0]
            slot[0] = move


def _bump_history(color: bool, move: chess.Move, depth: int) -> None:
    """Höj history-poängen för ett tyst drag som gav beta-cutoff."""
    table = _HISTORY[color]
    idx = (move.from_square << 6) | move.to_square
    new = table[idx] + depth * depth
    if new >= _HISTORY_MAX:  # skala ned allt så relativ ordning bevaras
        table[:] = [v >> 1 for v in table]
        new >>= 1
    table[idx] = new


def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
    """Zugzwang-skydd för null move: kräv minst en icke-bondepjäs."""
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if board.pieces(piece_type, color):
            return True
    return False


# ---------------------------------------------------------------------------
# Quiescence Search
# ---------------------------------------------------------------------------

def _quiescence(
    board: chess.Board,
    key: int,
    alpha: float,
    beta: float,
    model,
    device: str,
    perspective: int,
    qply: int,
) -> float:
    """Sök endast slag tills ställningen är 'lugn' för att undvika
    horisonteffekten. Delta pruning + gallring av förlorande slag håller
    antalet NN-anrop nere."""
    # Stand pat: nuvarande ställnings NN-värde ur sidan-vid-dragets perspektiv.
    # Terminala noder (matt/patt) har redan hanterats av föräldern/dragloopen;
    # kvarstår otillräckligt material som snabbkollas billigt här.
    if board.is_insufficient_material():
        return 0.0
    stand_pat = perspective * _evaluate_nn(board, key, model, device)

    if qply >= QSEARCH_MAX_PLY:
        return stand_pat
    if stand_pat >= beta:
        return stand_pat  # fail-soft
    if stand_pat > alpha:
        alpha = stand_pat

    best = stand_pat
    for move in _ordered_captures(board):
        # Delta pruning: även med offret + marginal kan draget inte höja alpha.
        if stand_pat + _victim_value(board, move) + DELTA_MARGIN <= alpha:
            continue
        # Gallra slag som ser klart förlorande ut (grov SEE-approximation).
        if _is_losing_capture(board, move):
            continue

        board.push(move)
        child_key = position_key(board)
        if board.is_checkmate():
            score = MATE_VALUE  # vi mattade motståndaren
        else:
            score = -_quiescence(board, child_key, -beta, -alpha, model,
                                 device, -perspective, qply + 1)
        board.pop()

        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break  # beta-cutoff
    return best


# ---------------------------------------------------------------------------
# Negamax med alpha-beta, PVS, LMR och LMP
# ---------------------------------------------------------------------------

def _negamax(
    board: chess.Board,
    key: int,
    depth: int,
    alpha: float,
    beta: float,
    model,
    device: str,
    perspective: int,
    history: set[int],
    killers: list[list[chess.Move | None]],
    ply: int,
) -> float:
    """Rekursiv negamax. Returnerar poäng ur sidan-vid-dragets perspektiv.
    Endast push()/pop() — aldrig board.copy(). `key` är nodens Zobrist-hash,
    beräknad av föräldern (en hashning per nod totalt)."""
    alpha_orig = alpha

    # --- TT-uppslag ---------------------------------------------------------
    tt_entry = _TT.get(key)
    tt_move: chess.Move | None = None
    if tt_entry is not None:
        tt_depth, tt_raw_score, tt_flag, tt_move = tt_entry
        if tt_depth >= depth:
            tt_score = _tt_score_from(tt_raw_score, ply)
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER:
                if tt_score > alpha:
                    alpha = tt_score
            elif tt_flag == TT_UPPER:
                if tt_score < beta:
                    beta = tt_score
            if alpha >= beta:
                return tt_score

    if board.is_insufficient_material():
        return 0.0

    in_check = board.is_check()

    if depth <= 0 or ply >= MAX_PLY:
        # Matt/patt vid horisonten fångas via evaluate-vägen i quiescence:
        # utan legala drag finns heller inga slag, så stand pat gäller — men
        # matt/patt måste kollas explicit eftersom NN inte kan avgöra det.
        if not any(board.legal_moves):
            return (-MATE_VALUE + ply) if in_check else 0.0
        return _quiescence(board, key, alpha, beta, model, device,
                           perspective, 0)

    # --- Null Move Pruning ------------------------------------------------------
    # Klassisk variant: "stå över" ett drag och sök grundare med nollfönster.
    # Kräver ingen statisk evaluering (ingen extra NN-forward-pass), till
    # skillnad från static null move / reverse futility som medvetet undviks.
    if (
        depth >= NULL_MOVE_MIN_DEPTH
        and not in_check
        and beta < MATE_THRESHOLD
        and _has_non_pawn_material(board, board.turn)
    ):
        board.push(chess.Move.null())
        null_key = position_key(board)
        null_score = -_negamax(
            board, null_key, depth - 1 - NULL_MOVE_REDUCTION, -beta,
            -beta + 1.0, model, device, -perspective, history, killers,
            ply + 1,
        )
        board.pop()
        if null_score >= beta:
            return null_score if null_score < MATE_THRESHOLD else beta

    # --- Dragloop (PVS + LMR + LMP) -----------------------------------------------
    killer_slot = killers[ply] if ply < MAX_PLY else (None, None)
    moves = _ordered_moves(board, tt_move, killer_slot[0], killer_slot[1])

    if not moves:  # matt eller patt — inga separata is_checkmate/is_stalemate
        return (-MATE_VALUE + ply) if in_check else 0.0

    lmp_limit = LMP_BASE + 4 * depth if (
        depth <= LMP_MAX_DEPTH and not in_check and alpha > -MATE_THRESHOLD
    ) else 1 << 30

    best = -INF
    best_move_here: chess.Move | None = None

    for move_index, move in enumerate(moves):
        is_capture = board.is_capture(move)
        is_promotion = move.promotion is not None
        is_quiet = not is_capture and not is_promotion

        # Late Move Pruning: hoppa över sena tysta drag på grunda noder —
        # kräver inget NN-anrop och krymper trädet kraftigt.
        if (
            is_quiet
            and move_index >= lmp_limit
            and best > -MATE_THRESHOLD
            and move != killer_slot[0]
            and move != killer_slot[1]
            and not board.gives_check(move)  # lat: endast för LMP-kandidater
        ):
            continue

        # LMR-beslut: gives_check() är dyrt — anropa det LAT, endast när
        # alla billiga villkor redan är uppfyllda.
        reduction = 0
        if (
            depth >= LMR_MIN_DEPTH
            and move_index >= LMR_MOVE_THRESHOLD
            and is_quiet
            and not in_check
            and not board.gives_check(move)
        ):
            # Djup-/index-beroende reduktion: sena drag på djupa noder
            # reduceras mer (re-search räddar felbedömningar).
            reduction = 1
            if move_index >= 6:
                reduction = 2
            if depth >= 5 and move_index >= 12:
                reduction = 3
            if reduction > depth - 1:
                reduction = depth - 1

        board.push(move)
        child_key = position_key(board)

        # Straffa återgång till redan besökt ställning (dragupprepning).
        penalty = REPETITION_PENALTY if child_key in history else 0.0

        if board.is_checkmate():
            score = MATE_VALUE - (ply + 1)  # direkt matt — inget barnanrop
        elif move_index == 0:
            # PV-draget: fullt fönster, fullt djup.
            score = -_negamax(
                board, child_key, depth - 1, -beta, -alpha, model, device,
                -perspective, history, killers, ply + 1,
            ) - penalty
        else:
            # Nollfönster (ev. reducerat djup).
            score = -_negamax(
                board, child_key, depth - 1 - reduction, -alpha - 1.0,
                -alpha, model, device, -perspective, history, killers,
                ply + 1,
            ) - penalty
            # LMR-re-search: reducerad sökning slog alpha -> fullt djup, nollfönster.
            if reduction and score > alpha:
                score = -_negamax(
                    board, child_key, depth - 1, -alpha - 1.0, -alpha,
                    model, device, -perspective, history, killers, ply + 1,
                ) - penalty
            # PVS-re-search: nollfönstret slog alpha inom (alpha, beta)
            # -> fullständig sökning med fullt fönster.
            if alpha < score < beta:
                score = -_negamax(
                    board, child_key, depth - 1, -beta, -alpha, model,
                    device, -perspective, history, killers, ply + 1,
                ) - penalty

        board.pop()

        if score > best:
            best = score
            best_move_here = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    # Beta-cutoff: tysta drag blir killers + history-poäng.
                    if is_quiet:
                        _store_killer(killers, ply, move)
                        _bump_history(board.turn, move, depth)
                    break

    # --- TT-lagring -----------------------------------------------------------
    if best <= alpha_orig:
        flag = TT_UPPER
    elif best >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    _tt_store(key, depth, best, flag, best_move_here, ply)

    return best


# ---------------------------------------------------------------------------
# Rot: batch-evaluering + iterativ fördjupning
# ---------------------------------------------------------------------------

def _batch_order_root_moves(
    board: chess.Board,
    legal_moves: list[chess.Move],
    model,
    device: str,
    perspective: int,
    history: set[int],
) -> tuple[list[tuple[float, chess.Move]], list[int]]:
    """Evaluera alla rotdrag i EN NN-batch (via push/pop, inga board.copy())
    och returnera (poäng, drag) sorterat bäst-först ur sidan-vid-dragets
    perspektiv, plus barnens Zobrist-nycklar i samma ordning."""
    n = len(legal_moves)
    keys: list[int] = []
    white_evals = [0.0] * n
    to_eval_idx: list[int] = []
    to_eval_tensors: list[torch.Tensor] = []

    for i, move in enumerate(legal_moves):
        board.push(move)
        key = position_key(board)
        keys.append(key)
        if board.is_checkmate():
            white_evals[i] = -MATE_VALUE if board.turn == chess.WHITE else MATE_VALUE
        elif board.is_stalemate() or board.is_insufficient_material():
            white_evals[i] = 0.0
        else:
            cached = _EVAL_CACHE.get(key)
            if cached is not None:
                white_evals[i] = cached
            else:
                to_eval_idx.append(i)
                to_eval_tensors.append(board_to_tensor(board))
        board.pop()

    if to_eval_tensors:
        with torch.no_grad():
            batch = torch.stack(to_eval_tensors).to(device)
            values = model(batch).squeeze(-1).tolist()
        if isinstance(values, float):
            values = [values]
        for idx, v in zip(to_eval_idx, values):
            white_evals[idx] = v
            _evict_eval_cache()
            _EVAL_CACHE[keys[idx]] = v

    scored: list[tuple[float, chess.Move]] = []
    for move, key, we in zip(legal_moves, keys, white_evals):
        s = perspective * we
        if key in history:
            s -= REPETITION_PENALTY
        scored.append((s, move))

    order = sorted(range(n), key=lambda i: scored[i][0], reverse=True)
    return [scored[i] for i in order], [keys[i] for i in order]


def find_best_move(
    board: chess.Board,
    model,
    depth: int = 2,
    device: str = "cpu",
    history: list[int] | set[int] | None = None,
    max_seconds: float | None = None,
) -> tuple[chess.Move | None, float]:
    """Hitta bästa draget med iterativ fördjupning från djup 1 till `depth`.

    Args:
        board:       aktuell ställning (muteras endast via push/pop).
        model:       ChessEvaluatorNN-instans.
        depth:       maximalt sökdjup.
        device:      "cpu" eller "cuda".
        history:     Zobrist-nycklar för tidigare besökta ställningar
                     (för dragupprepnings-straff).
        max_seconds: mjuk tidsgräns — påbörjar inte en ny iteration som
                     förväntas spränga budgeten.

    Returns:
        (bästa drag eller None, poäng ur sidan-vid-dragets perspektiv).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0

    history_set: set[int] = set(history) if history else set()
    perspective = 1 if board.turn == chess.WHITE else -1

    try:
        model.to(device)
        model.eval()
    except Exception:
        pass  # tolerera enkla callables/mockar utan .to()/.eval()

    # Killer moves: förallokerad 2D-struktur, MAX_PLY x 2.
    killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]

    root_key = position_key(board)

    # --- Steg 1: batchad grundordning av rotdragen -----------------------------
    scored, child_keys = _batch_order_root_moves(
        board, legal_moves, model, device, perspective, history_set)
    ordered_moves = [m for _, m in scored]
    move_keys = {m: k for m, k in zip(ordered_moves, child_keys)}
    best_score, best_move = scored[0]  # baslinje om djup 1 aldrig hinner köras

    # Senast kända poäng per rotdrag — seedas med NN-batchens snabbevaluering
    # och uppdateras varje iteration. Används för rotgallringen.
    last_scores: dict[chess.Move, float] = {m: s for s, m in scored}

    # --- Steg 2: iterativ fördjupning med PVS i roten --------------------------
    start_time = time.time()
    last_iter_time: float | None = None

    for current_depth in range(1, depth + 1):
        if max_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                break
            if last_iter_time is not None and \
                    elapsed + last_iter_time * ITER_TIME_GROWTH_ESTIMATE > max_seconds:
                break

        iter_start = time.time()
        alpha, beta = -INF, INF

        # Sätt föregående iterations bästa drag (via TT) först i ordningen.
        tt_entry = _TT.get(root_key)
        tt_move = tt_entry[3] if tt_entry is not None else None
        if tt_move is not None and tt_move in move_keys:
            search_order = [tt_move] + [m for m in ordered_moves if m != tt_move]
        else:
            search_order = ordered_moves

        # --- Rotgallring: hoppa över drag vi redan VET är dåliga -----------------
        # Poängen från föregående iteration (eller NN-baslinjen) jämförs mot
        # bästa draget; allt under marginalen söks inte djupare. Minst
        # ROOT_MIN_MOVES kandidater behålls alltid, och gallringen stängs av
        # när mattpoäng är inblandade (då är poängskalan inte jämförbar).
        if (
            current_depth >= ROOT_PRUNE_MIN_DEPTH
            and len(search_order) > ROOT_MIN_MOVES
            and abs(best_score) < MATE_THRESHOLD
        ):
            margin = max(
                ROOT_PRUNE_MARGIN_BASE
                - ROOT_PRUNE_MARGIN_STEP * (current_depth - ROOT_PRUNE_MIN_DEPTH),
                ROOT_PRUNE_MARGIN_MIN,
            )
            cutoff = max(last_scores.values()) - margin
            pruned = [m for m in search_order if last_scores[m] >= cutoff]
            if len(pruned) < ROOT_MIN_MOVES:
                # Fyll upp med de bäst rankade av de bortgallrade.
                rest = [m for m in search_order if m not in pruned]
                rest.sort(key=lambda m: last_scores[m], reverse=True)
                pruned.extend(rest[: ROOT_MIN_MOVES - len(pruned)])
                pruned.sort(key=lambda m: search_order.index(m))
            search_order = pruned

        iter_best_move: chess.Move | None = None
        iter_best_score = -INF

        for move_index, move in enumerate(search_order):
            child_key = move_keys[move]
            board.push(move)
            penalty = REPETITION_PENALTY if child_key in history_set else 0.0

            if move_index == 0:
                score = -_negamax(
                    board, child_key, current_depth - 1, -beta, -alpha,
                    model, device, -perspective, history_set, killers, 1,
                ) - penalty
            else:
                score = -_negamax(
                    board, child_key, current_depth - 1, -alpha - 1.0,
                    -alpha, model, device, -perspective, history_set,
                    killers, 1,
                ) - penalty
                if alpha < score < beta:
                    score = -_negamax(
                        board, child_key, current_depth - 1, -beta, -alpha,
                        model, device, -perspective, history_set, killers, 1,
                    ) - penalty

            board.pop()

            # Obs: för icke-PV-drag är nollfönsterpoängen en övre gräns —
            # fullt tillräcklig för gallring (verklig poäng är aldrig högre).
            last_scores[move] = score

            if score > iter_best_score:
                iter_best_score = score
                iter_best_move = move
            if iter_best_score > alpha:
                alpha = iter_best_score

        if iter_best_move is not None:
            best_move, best_score = iter_best_move, iter_best_score
            _tt_store(root_key, current_depth, best_score, TT_EXACT, best_move, 0)
            # Sortera om: nya bästa draget först inför nästa iteration.
            ordered_moves = [best_move] + [m for m in ordered_moves if m != best_move]

        last_iter_time = time.time() - iter_start

    return best_move, best_score
