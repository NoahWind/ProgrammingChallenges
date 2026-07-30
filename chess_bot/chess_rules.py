# -*- coding: utf-8 -*-
"""
Schacklogik - pjäsdrag, hot, schack, schackmatt och patt.

Vad som fixades jämfört med originalet:
- Rockad-rättigheter/en passant-ruta hålls i ett GameState-objekt istället
  för lösa globala variabler. Originalet hade EN gemensam
  "kungen har flyttat"-flagga som gällde BÅDA färgerna - så fort ena
  sidans kung flyttade blockerades rockad för motståndaren också.
- Rockad kontrollerar nu: rutorna är tomma OCH inte hotade, samt att
  kungen inte redan står i schack (man fick tidigare rockera genom/
  ur/in i schack).
- En passant lagras som en enda ruta (eller None) istället för en lista,
  och legalitetskontrollen tar nu faktiskt bort den slagna bonden när
  den simulerar draget (annars kunde en passant-drag som avslöjar
  schack räknas som lagligt).
- apply_move() uppdaterar alla flaggor (kung/torn flyttad, en passant-
  ruta, rockad-tornets flytt, promotion) på ETT ställe, så man slipper
  sprida ut och glömma uppdateringar i motorkoden.
- Effektivare: kungens position cachas/skickas in istället för att
  scanna hela brädet (64 rutor) för varje enskilt pseudo-drag i
  get_all_legal_moves - det gav tidigare en extra O(64) per drag helt
  i onödan.
"""

import time

from dataclasses import dataclass
from typing import Optional, Tuple, List

WHITE_PIECES = frozenset('♙♘♗♖♕♔')
BLACK_PIECES = frozenset('♟♞♝♜♛♚')

_ROOK_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_BISHOP_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_QUEEN_DIRS = _ROOK_DIRS + _BISHOP_DIRS
_KNIGHT_DELTAS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1))
_KING_DELTAS = ((-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1))

PIECE_VALUES_MAP = {
    '♟': -1, '♞': -3, '♝': -3, '♜': -5, '♛': -9, '♚': -10000,
    '♙': 1,  '♘': 3,  '♗': 3,  '♖': 5,  '♕': 9,  '♔': 10000,
    ' ': 0
}

# ---------------------------------------------------------------------
# Speltillstånd (ersätter de gamla globala variablerna)
# ---------------------------------------------------------------------
@dataclass
class GameState:
    white_king_moved: bool = False
    black_king_moved: bool = False
    rook_a1_moved: bool = False
    rook_h1_moved: bool = False
    rook_a8_moved: bool = False
    rook_h8_moved: bool = False
    en_passant_target: Optional[Tuple[int, int]] = None
    
    # NYA FÄLT FÖR OPTIMERING
    white_pieces: set = None
    black_pieces: set = None
    white_king_pos: Tuple[int, int] = None
    black_king_pos: Tuple[int, int] = None
    material_score: int = 0
    half_move_clock: int = 0

    def init_pieces(self, board):
        """Körs en gång vid spelets start för att hitta alla pjäser."""
        self.white_pieces = set()
        self.black_pieces = set()
        self.material_score = 0
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p in WHITE_PIECES:
                    self.white_pieces.add((r, c))
                    if p == '♔': self.white_king_pos = (r, c)
                    self.material_score += PIECE_VALUES_MAP[p]
                elif p in BLACK_PIECES:
                    self.black_pieces.add((r, c))
                    if p == '♚': self.black_king_pos = (r, c)
                    self.material_score += PIECE_VALUES_MAP[p]

    def king_moved(self, color: str) -> bool:
        return self.white_king_moved if color == 'white' else self.black_king_moved

    def rook_moved(self, color: str, side: str) -> bool:
        if color == 'white':
            return self.rook_h1_moved if side == 'k' else self.rook_a1_moved
        return self.rook_h8_moved if side == 'k' else self.rook_a8_moved
def get_piece_color(piece: str) -> Optional[str]:
    if piece in WHITE_PIECES:
        return 'white'
    if piece in BLACK_PIECES:
        return 'black'
    return None


# ---------------------------------------------------------------------
# Pjäsposition
# ---------------------------------------------------------------------

def get_location_of_all_pieces(board):
    pieces_white = []
    pieces_black = []
    for i in range(8):
        row = board[i]
        for j in range(8):
            piece = row[j]
            if piece == ' ':
                continue
            if piece in WHITE_PIECES:
                pieces_white.append((piece, i, j))
            else:
                pieces_black.append((piece, i, j))
    return pieces_white, pieces_black


def find_king(board, color) -> Optional[Tuple[int, int]]:
    king = '♔' if color == 'white' else '♚'
    for i in range(8):
        row = board[i]
        for j in range(8):
            if row[j] == king:
                return (i, j)
    return None


# ---------------------------------------------------------------------
# Pseudo-lagliga drag per pjästyp
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Pseudo-lagliga drag per pjästyp (OPTIMERAD)
# ---------------------------------------------------------------------

def _add_sliding_moves(board, row, col, own_set, dirs, raw_moves, captures_only):
    for dr, dc in dirs:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            sq = board[r][c]
            if sq == ' ':
                if not captures_only:
                    raw_moves.append(((row, col), (r, c), False, False))
            elif sq in own_set:
                break
            else:
                raw_moves.append(((row, col), (r, c), False, False))
                break
            r += dr
            c += dc

def add_knight_moves(board, row, col, color, raw_moves, captures_only):
    own = WHITE_PIECES if color == 'white' else BLACK_PIECES
    for dr, dc in _KNIGHT_DELTAS:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            sq = board[r][c]
            if sq not in own:
                if not captures_only or sq != ' ':
                    raw_moves.append(((row, col), (r, c), False, False))

def add_king_moves(board, row, col, color, raw_moves, captures_only, state: GameState):
    own = WHITE_PIECES if color == 'white' else BLACK_PIECES
    for dr, dc in _KING_DELTAS:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            sq = board[r][c]
            if sq not in own:
                if not captures_only or sq != ' ':
                    raw_moves.append(((row, col), (r, c), False, False))

    if not captures_only and not state.king_moved(color):
        home_row = 7 if color == 'white' else 0
        if row == home_row and col == 4:
            if not square_attacked(board, row, col, color):
                # Kungsidan
                if not state.rook_moved(color, 'k') and board[row][5] == ' ' and board[row][6] == ' ':
                    if not square_attacked(board, row, 5, color) and not square_attacked(board, row, 6, color):
                        raw_moves.append(((row, col), (row, 6), False, True))
                # Damsidan
                if not state.rook_moved(color, 'q') and board[row][1] == ' ' and board[row][2] == ' ' and board[row][3] == ' ':
                    if not square_attacked(board, row, 3, color) and not square_attacked(board, row, 2, color):
                        raw_moves.append(((row, col), (row, 2), False, True))

def add_pawn_moves(board, row, col, color, raw_moves, captures_only, state: GameState):
    if color == 'white':
        direction, start_row, opp_set, er_row = -1, 6, BLACK_PIECES, 3
    else:
        direction, start_row, opp_set, er_row = 1, 1, WHITE_PIECES, 4

    nr = row + direction
    if 0 <= nr < 8:
        if board[nr][col] == ' ':
            # Kolla om draget leder till sista raden (förvandling)
            is_promotion = (nr == 0 or nr == 7)
            
            # Tillåt framryckning om vi antingen vill ha alla drag, 
            # ELLER om vi bara vill ha captures men draget är en förvandling.
            if not captures_only or is_promotion:
                raw_moves.append(((row, col), (nr, col), False, False))
                
            # Dubbelsteg är aldrig förvandling, genereras bara när captures_only=False
            if not captures_only and row == start_row and board[row + 2 * direction][col] == ' ':
                raw_moves.append(((row, col), (row + 2 * direction, col), False, False))
                
        for dc in (-1, 1):
            nc = col + dc
            if 0 <= nc < 8 and board[nr][nc] in opp_set:
                raw_moves.append(((row, col), (nr, nc), False, False))

    # En passant
    if state.en_passant_target and row == er_row:
        tr, tc = state.en_passant_target
        if tr == nr and abs(tc - col) == 1:
            raw_moves.append(((row, col), (tr, tc), True, False))

# ---------------------------------------------------------------------
# Hot / schack (Behåll exakt som den är i din kod!)
# ---------------------------------------------------------------------

_ENEMY_ROOKLIKE_WHITE = frozenset('♜♛')
_ENEMY_BISHOPLIKE_WHITE = frozenset('♝♛')
_ENEMY_ROOKLIKE_BLACK = frozenset('♖♕')
_ENEMY_BISHOPLIKE_BLACK = frozenset('♗♕')

def square_attacked(board, row, col, defender_color):
    # BEHÅLL DIN KOD HÄR (den är redan snabb)
    if defender_color == 'white':
        rooklike, bishoplike = _ENEMY_ROOKLIKE_WHITE, _ENEMY_BISHOPLIKE_WHITE
        enemy_knight, enemy_king, enemy_pawn = '♞', '♚', '♟'
        pawn_dr = -1
    else:
        rooklike, bishoplike = _ENEMY_ROOKLIKE_BLACK, _ENEMY_BISHOPLIKE_BLACK
        enemy_knight, enemy_king, enemy_pawn = '♘', '♔', '♙'
        pawn_dr = 1

    pr = row + pawn_dr
    if 0 <= pr < 8:
        if col - 1 >= 0 and board[pr][col - 1] == enemy_pawn:
            return True
        if col + 1 < 8 and board[pr][col + 1] == enemy_pawn:
            return True

    for dr, dc in _KNIGHT_DELTAS:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == enemy_knight:
            return True

    for dr, dc in _KING_DELTAS:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == enemy_king:
            return True

    for dr, dc in _ROOK_DIRS:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            sq = board[r][c]
            if sq != ' ':
                if sq in rooklike:
                    return True
                break
            r += dr
            c += dc

    for dr, dc in _BISHOP_DIRS:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            sq = board[r][c]
            if sq != ' ':
                if sq in bishoplike:
                    return True
                break
            r += dr
            c += dc

    return False

def is_check(board, color, state: GameState, king_pos: Optional[Tuple[int, int]] = None):
    if king_pos is None:
        # Hämtar positionen direkt från vårt nya snabba state
        king_pos = state.white_king_pos if color == 'white' else state.black_king_pos
        if king_pos is None:
            return False
    return square_attacked(board, king_pos[0], king_pos[1], color)


# ---------------------------------------------------------------------
# Laglig-drag-generering (OPTIMERAD)
# ---------------------------------------------------------------------
def get_all_legal_moves(color, board, state: GameState, captures_only=False):
    own_pieces_set = state.white_pieces if color == 'white' else state.black_pieces
    own_pieces = WHITE_PIECES if color == 'white' else BLACK_PIECES
    king_pos = state.white_king_pos if color == 'white' else state.black_king_pos

    if not king_pos:
        return []

    raw_moves = []

    for r, c in own_pieces_set:
        p = board[r][c]
        if p in ('♙', '♟'):
            add_pawn_moves(board, r, c, color, raw_moves, captures_only, state)
        elif p in ('♖', '♜'):
            _add_sliding_moves(board, r, c, own_pieces, _ROOK_DIRS, raw_moves, captures_only)
        elif p in ('♘', '♞'):
            add_knight_moves(board, r, c, color, raw_moves, captures_only)
        elif p in ('♗', '♝'):
            _add_sliding_moves(board, r, c, own_pieces, _BISHOP_DIRS, raw_moves, captures_only)
        elif p in ('♕', '♛'):
            _add_sliding_moves(board, r, c, own_pieces, _QUEEN_DIRS, raw_moves, captures_only)
        elif p in ('♔', '♚'):
            add_king_moves(board, r, c, color, raw_moves, captures_only, state)

    legal = []
    for (sr, sc), (er, ec), is_ep, _is_castle in raw_moves:
        record = apply_move(board, state, (sr, sc), (er, ec))
        try:
            cur_king_pos = (er, ec) if board[er][ec] in ('♔', '♚') else king_pos
            in_check = is_check(board, color, state, cur_king_pos)
        finally:
            undo_move(board, state, record)

        if not in_check:
            legal.append(((sr, sc), (er, ec)))

    return legal

def is_checkmate(board, color, state: GameState, king_pos: Optional[Tuple[int, int]] = None):
    if not is_check(board, color, state, king_pos):
        return False
    return len(get_all_legal_moves(color, board, state)) == 0


def is_stalemate(board, color, state: GameState):
    if is_check(board, color, state):
        return False
    return len(get_all_legal_moves(color, board, state)) == 0


def apply_move(board, state: GameState, start, end, promotion='queen'):
    sr, sc = start
    er, ec = end
    piece = board[sr][sc]
    color = get_piece_color(piece)

    # 1. Definiera variabler direkt
    is_pawn = piece in ('♙', '♟')
    is_king = piece in ('♔', '♚')

    # 2. Ta ögonblicksbild av state
    prev_state = (
        state.white_king_moved, state.black_king_moved,
        state.rook_a1_moved, state.rook_h1_moved,
        state.rook_a8_moved, state.rook_h8_moved,
        state.en_passant_target,
        state.material_score,
        state.half_move_clock
    )

    captured = board[er][ec]
    captured_square = (er, ec)
    en_passant_capture = False

    # En passant-slag
    if is_pawn and captured == ' ' and sc != ec and state.en_passant_target == (er, ec):
        captured_square = (sr, ec)
        captured = board[sr][ec]
        board[sr][ec] = ' '
        en_passant_capture = True

    # Rockad
    rook_move = None
    if is_king and abs(ec - sc) == 2:
        row = sr
        if ec == 6:  # kungsidan
            rook_move = ((row, 7), (row, 5))
        else:  # damsidan
            rook_move = ((row, 0), (row, 3))
        (rfr, rfc), (rtr, rtc) = rook_move
        board[rtr][rtc] = board[rfr][rfc]
        board[rfr][rfc] = ' '

    board[er][ec] = piece
    board[sr][sc] = ' '

    # Promotion
    promoted = False
    if is_pawn and (er == 0 or er == 7):
        promo_map_white = {'queen': '♕', 'rook': '♖', 'bishop': '♗', 'knight': '♘'}
        promo_map_black = {'queen': '♛', 'rook': '♜', 'bishop': '♝', 'knight': '♞'}
        board[er][ec] = promo_map_white[promotion] if color == 'white' else promo_map_black[promotion]
        promoted = True

    # Uppdatera kung/torn-flaggor
    if is_king:
        if color == 'white': state.white_king_moved = True
        else: state.black_king_moved = True
    if piece in ('♖', '♜'):
        if (sr, sc) == (7, 0): state.rook_a1_moved = True
        elif (sr, sc) == (7, 7): state.rook_h1_moved = True
        elif (sr, sc) == (0, 0): state.rook_a8_moved = True
        elif (sr, sc) == (0, 7): state.rook_h8_moved = True
    if (er, ec) == (7, 0): state.rook_a1_moved = True
    elif (er, ec) == (7, 7): state.rook_h1_moved = True
    elif (er, ec) == (0, 0): state.rook_a8_moved = True
    elif (er, ec) == (0, 7): state.rook_h8_moved = True

    # En passant-target
    if is_pawn and abs(er - sr) == 2:
        state.en_passant_target = ((er + sr) // 2, ec)
    else:
        state.en_passant_target = None

    state.half_move_clock = prev_state[8] + 1 if piece not in ('♙', '♟') and captured == ' ' else 0

    # Inkrementell uppdatering
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    opp_pieces = state.black_pieces if color == 'white' else state.white_pieces

    own_pieces.remove(start)
    own_pieces.add(end)

    if is_king:
        if color == 'white': state.white_king_pos = end
        else: state.black_king_pos = end

    if captured != ' ':
        opp_pieces.remove(captured_square)
        state.material_score -= PIECE_VALUES_MAP[captured]

    if promoted:
        state.material_score -= PIECE_VALUES_MAP[piece]
        state.material_score += PIECE_VALUES_MAP[board[er][ec]]

    if rook_move:
        (rfr, rfc), (rtr, rtc) = rook_move
        own_pieces.remove((rfr, rfc))
        own_pieces.add((rtr, rtc))

    return {
        'start': start, 'end': end, 'piece': piece, 'color': color,
        'captured': captured, 'captured_square': captured_square,
        'en_passant_capture': en_passant_capture, 'rook_move': rook_move,
        'promoted': promoted, 'prev_state': prev_state,
    }


def undo_move(board, state: GameState, record):
    """Ångrar exakt det drag som gav upphov till `record`
    (returvärdet från apply_move). Återställer både bräde och state."""
    sr, sc = record['start']
    er, ec = record['end']

    # Lägg tillbaka pjäsen på startrutan
    board[sr][sc] = record['piece']
    board[er][ec] = ' '

    # Om det var en rockad, flytta tillbaka tornet
    if record['rook_move']:
        (rfr, rfc), (rtr, rtc) = record['rook_move']
        board[rfr][rfc] = board[rtr][rtc]
        board[rtr][rtc] = ' '

    # Återställ slagen pjäs
    csr, csc = record['captured_square']
    board[csr][csc] = record['captured']

    # Återställ state-flaggor OCH material_score (8 st värden totalt)
    (state.white_king_moved, state.black_king_moved,
     state.rook_a1_moved, state.rook_h1_moved,
     state.rook_a8_moved, state.rook_h8_moved,
     state.en_passant_target,
     state.material_score,
     state.half_move_clock) = record['prev_state']
    
    # Återställning av inkrementella positioner
    color = record['color']
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    opp_pieces = state.black_pieces if color == 'white' else state.white_pieces

    # Flytta tillbaka egen pjäs
    own_pieces.remove(record['end'])
    own_pieces.add(record['start'])

    # Om kungen flyttades, återställ dess sparade position
    if record['piece'] in ('♔', '♚'):
        if color == 'white': state.white_king_pos = record['start']
        else: state.black_king_pos = record['start']

    # Om en pjäs slogs, lägg tillbaka den i mängden
    if record['captured'] != ' ':
        opp_pieces.add(record['captured_square'])

    # Om rockad skedde, flytta tillbaka tornet i mängden
    if record['rook_move']:
        (rfr, rfc), (rtr, rtc) = record['rook_move']
        own_pieces.remove((rtr, rtc))
        own_pieces.add((rfr, rfc))

def get_position_hash(board, color, state):
    """Skapar en unik nyckel för en ställning (bräde, tur, och state-rättigheter)."""
    state_key = (
        state.white_king_moved, state.black_king_moved,
        state.rook_a1_moved, state.rook_h1_moved,
        state.rook_a8_moved, state.rook_h8_moved,
        state.en_passant_target
    )
    # Konverterar brädet till en tuple av tuples så att det är hashbart (immutable)
    board_tuple = tuple(tuple(row) for row in board)
    return (board_tuple, color, state_key)