import time
from chess_rules import get_all_legal_moves, get_location_of_all_pieces, is_check, apply_move, undo_move, is_stalemate, get_position_hash
# ==========================================
# ÄNDRAT jämfört med föregående version:
#
# - Importerar chess_rules istället för get_leagle_moves. Alla anrop till
#   get_all_legal_moves tar nu (color, board, state) - inte längre en
#   tom fen-sträng som första argument.
# - _apply_move/_undo_move är borttagna. Vi använder chess_rules
#   apply_move()/undo_move() istället, som (till skillnad från de gamla)
#   faktiskt flyttar tornet vid rockad och tar bort rätt bonde vid en
#   passant - och är O(1) att ångra tack vare move-record.
# - is_check/is_checkmate är inte längre egna implementationer här -
#   de importeras från chess_rules, som gör dem UTAN att generera alla
#   motståndarens drag (det gamla is_check gjorde full draggenerering
#   bara för att se om kungens ruta var ett mål - onödigt dyrt).
# - forced_mate är helt borttagen. Den anropades inuti evaluate_board,
#   dvs vid VARJE löv i sökträdet, och gjorde där en egen rekursiv
#   sökning 3 halvdrag djupt - det multiplicerade hela trädets storlek
#   med en extra exponentiell faktor. Riktig mattdetektering sker redan
#   korrekt i minimax/quiescence_search via "inga lagliga drag + schack".
# - evaluate_board() tar nu emot `state` (krävs av get_all_legal_moves),
#   och anropar inte längre get_all_legal_moves alls - de gamla
#   corner-control- och is_checkmate-kollen i evaluate_board gjorde
#   full draggenerering för BÅDA färgerna vid varje enda utvärdering,
#   vilket är den absolut hetaste koden i hela motorn. De heuristikerna
#   (värda 0.2-100000 poäng) är borttagna till förmån för hastighet;
#   riktig matt/patt hanteras ändå korrekt av minimax/quiescence_search.
# - Dubblerad knight_on_the_rim-bestraffning i evaluate_board borttagen
#   (fanns kopierad två gånger av misstag).
# - Transpositionstabellens nyckel innehöll bara brädet - inte vems tur
#   det är eller rockad-/en passant-rättigheter. Två likadana brädbilder
#   med olika rättigheter gav då samma (potentiellt felaktiga) cache-
#   träff. Nyckeln inkluderar nu allt relevant state.
# - quiescence_search kollade aldrig om sidan i draget är schackmatt/
#   patt innan den föll tillbaka på stand-pat-värdet - fixat, annars kan
#   motorn missa matt precis vid sökhorisonten.
# - En passant-slag räknades inte som "slag" i quiescence-filtreringen
#   (målrutan är tom vid en passant) - fixat med _is_capture_move().
# ==========================================

from different_evals import *

ENGINE_PARAMS ={
    "BREAKING_PAWN_CHAINS_BONUS": 0.0021,
    "CONTROL_CENTER_BONUS": 14.7545,
    "END_RATE": 0.0328,
    "KING_SAFETY_BONUS": 0.0004,
    "OPEN_RATE": 0.0199,
    "PASSED_PAWN_BONUS": 0.1252,
    "bishop_pair_bonus": 0.5911,
    "enemy_king_center_bonus": 0.1252,
    "enemy_king_corner_bonus": 0.2503,
    "hanging_piece_penalty": -557,
    "isolated_pawn_penalty": 0.1156,
    "knight_on_the_rim_penalty": 0.3939,
    "pawn_chain_bonus": 1.1622,
    "pieac_pos_bonus": 24.8262,
    "rook_open_file_bonus": 0.0,
    "squares_controlled_bonus": 0.0,
    "rook_on_seventh_rank_bonus": 0.0,


    }
def _has_any_pawns(board):
    return any(piece in ('♙', '♟') for row in board for piece in row)

def get_game_phase_weights(state, open_rate=0.6, end_rate=0.6):
    """
    Beräknar a (öppning), b (mittspel) och c (slutspel) baserat på pjäser kvar.
    
    - open_rate: Bestämmer hur snabbt öppningen fasas ut (högre värde = öppningen hänger med längre).
    - end_rate: Bestämmer när slutspelet kickar in (högre värde = slutspelet börjar tidigare).
    """
    total_pieces = len(state.white_pieces) + len(state.black_pieces)
    
    # Normalisera p från 1.0 (fullt bräde) till 0.0 (bara kungar kvar)
    p = max(0.0, min(1.0, (total_pieces - 2) / 30.0))
    
    # a = Öppningsvikt
    a = max(0.0, (p - (1.0 - open_rate)) / open_rate) if open_rate > 0 else 0.0
    
    # b = Mittspelsvikt (toppar i mitten av partiet)
    b = 1.0 - abs(p - 0.5) * 2
    
    # c = Slutspelsvikt
    c = max(0.0, (end_rate - p) / end_rate) if end_rate > 0 else 0.0
    
    return a, b, c

def rook_on_the_seventh_rank(board, color, state):
    """
    Returnerar True om en av spelarens torn är på den 7:e raden (för vit) eller 2:a raden (för svart).
    """
    target_rank = 6 if color == 'white' else 1
    rook_symbol = '♖' if color == 'white' else '♜'
    
    for col in range(8):
        if board[target_rank][col] == rook_symbol:
            return True
    return False


def evaluate_board(board, state, history, current_color, ENGINE_PARAMS=ENGINE_PARAMS):
    score = float(state.material_score)


    
    pos_key = get_position_hash(board, current_color, state)
    if (history.get(pos_key, 0) >= 2
            and abs(state.material_score) < 5
            and not _has_any_pawns(board)):
        return 0 
    moves = get_all_legal_moves(current_color, board, state)
    enemy_color = 'black' if current_color == 'white' else 'white'
    enemy_moves = get_all_legal_moves(enemy_color, board, state)
    
    
    a, b, c = get_game_phase_weights(state, ENGINE_PARAMS["OPEN_RATE"], ENGINE_PARAMS["END_RATE"])

    if state.half_move_clock > 80:
        if score < 0:
            score += (state.half_move_clock - 80) * 0.5 
        elif score > 0:
            score -= (state.half_move_clock - 80) * 0.5
    
    if is_king_safe(board, 'white', state): score += ENGINE_PARAMS["KING_SAFETY_BONUS"] * a + ENGINE_PARAMS["KING_SAFETY_BONUS"] * b + ENGINE_PARAMS["KING_SAFETY_BONUS"] * c
    if is_king_safe(board, 'black', state): score -= ENGINE_PARAMS["KING_SAFETY_BONUS"] * a + ENGINE_PARAMS["KING_SAFETY_BONUS"] * b + ENGINE_PARAMS["KING_SAFETY_BONUS"] * c

    if is_controling_center(board, 'white'): score += ENGINE_PARAMS["CONTROL_CENTER_BONUS"]  * a + ENGINE_PARAMS["CONTROL_CENTER_BONUS"] * b + ENGINE_PARAMS["CONTROL_CENTER_BONUS"] * c
    if is_controling_center(board, 'black'): score -= ENGINE_PARAMS["CONTROL_CENTER_BONUS"] * a + ENGINE_PARAMS["CONTROL_CENTER_BONUS"] * b + ENGINE_PARAMS["CONTROL_CENTER_BONUS"] * c

    if breakt_pawn_chains(board, 'white', state): score -= ENGINE_PARAMS["BREAKING_PAWN_CHAINS_BONUS"] * a + ENGINE_PARAMS["BREAKING_PAWN_CHAINS_BONUS"] * b + ENGINE_PARAMS["BREAKING_PAWN_CHAINS_BONUS"] * c
    if breakt_pawn_chains(board, 'black', state): score += ENGINE_PARAMS["BREAKING_PAWN_CHAINS_BONUS"] * a + ENGINE_PARAMS["BREAKING_PAWN_CHAINS_BONUS"] * b + ENGINE_PARAMS["BREAKING_PAWN_CHAINS_BONUS"] * c

    if bishop_pair(board, 'white', state): score += ENGINE_PARAMS["bishop_pair_bonus"] * a + ENGINE_PARAMS["bishop_pair_bonus"] * b + ENGINE_PARAMS["bishop_pair_bonus"] * c
    if bishop_pair(board, 'black', state): score -= ENGINE_PARAMS["bishop_pair_bonus"] * a + ENGINE_PARAMS["bishop_pair_bonus"] * b + ENGINE_PARAMS["bishop_pair_bonus"] * c

    if knight_on_the_rim(board, 'white', state): score -= ENGINE_PARAMS["knight_on_the_rim_penalty"] * a + ENGINE_PARAMS["knight_on_the_rim_penalty"] * b + ENGINE_PARAMS["knight_on_the_rim_penalty"] * c
    if knight_on_the_rim(board, 'black', state): score += ENGINE_PARAMS["knight_on_the_rim_penalty"] * a + ENGINE_PARAMS["knight_on_the_rim_penalty"] * b + ENGINE_PARAMS["knight_on_the_rim_penalty"] * c

    if isolated_pawn_penalty_or_doubling(board, 'white', state): score -= ENGINE_PARAMS["isolated_pawn_penalty"] * a + ENGINE_PARAMS["isolated_pawn_penalty"] * b + ENGINE_PARAMS["isolated_pawn_penalty"] * c
    if isolated_pawn_penalty_or_doubling(board, 'black', state): score += ENGINE_PARAMS["isolated_pawn_penalty"] * a + ENGINE_PARAMS["isolated_pawn_penalty"] * b + ENGINE_PARAMS["isolated_pawn_penalty"] * c

    if Rook_Open_Files(board, 'white', state): score += ENGINE_PARAMS["rook_open_file_bonus"] * a + ENGINE_PARAMS["rook_open_file_bonus"] * b + ENGINE_PARAMS["rook_open_file_bonus"] * c
    if Rook_Open_Files(board, 'black', state): score -= ENGINE_PARAMS["rook_open_file_bonus"] * a + ENGINE_PARAMS["rook_open_file_bonus"] * b + ENGINE_PARAMS["rook_open_file_bonus"] * c

    white_moves = moves if current_color == 'white' else enemy_moves
    black_moves = enemy_moves if current_color == 'white' else moves

    black_sqr = how_many_squares_do_i_control(board, 'black', state, black_moves)
    white_sqr = how_many_squares_do_i_control(board, 'white', state, white_moves)

    if black_sqr > 0: score -= ENGINE_PARAMS["squares_controlled_bonus"] * (black_sqr - white_sqr) * a + ENGINE_PARAMS["squares_controlled_bonus"] * (black_sqr - white_sqr) * b + ENGINE_PARAMS["squares_controlled_bonus"] * (black_sqr - white_sqr) * c
    if white_sqr > 0: score += ENGINE_PARAMS["squares_controlled_bonus"] * (white_sqr - black_sqr) * a + ENGINE_PARAMS["squares_controlled_bonus"] * (white_sqr - black_sqr) * b + ENGINE_PARAMS["squares_controlled_bonus"] * (white_sqr - black_sqr) * c

    pst_w, hang_w = evaluate_pieces_and_threats(board, state, 'white')
    pst_b, hang_b = evaluate_pieces_and_threats(board, state, 'black')

    score += pst_w * ENGINE_PARAMS["pieac_pos_bonus"] * a + pst_w * ENGINE_PARAMS["pieac_pos_bonus"] * b + pst_w * ENGINE_PARAMS["pieac_pos_bonus"] * c
    score += pst_b * ENGINE_PARAMS["pieac_pos_bonus"] * a + pst_b * ENGINE_PARAMS["pieac_pos_bonus"] * b + pst_b * ENGINE_PARAMS["pieac_pos_bonus"] * c
    score += hang_w * ENGINE_PARAMS["hanging_piece_penalty"] * a + hang_w * ENGINE_PARAMS["hanging_piece_penalty"] * b + hang_w * ENGINE_PARAMS["hanging_piece_penalty"] * c
    score += hang_b * ENGINE_PARAMS["hanging_piece_penalty"] * a + hang_b * ENGINE_PARAMS["hanging_piece_penalty"] * b + hang_b * ENGINE_PARAMS["hanging_piece_penalty"] * c


    if passed_pawn_score(board, 'white', state): score += a*ENGINE_PARAMS["PASSED_PAWN_BONUS"] + b*ENGINE_PARAMS["PASSED_PAWN_BONUS"] + c*ENGINE_PARAMS["PASSED_PAWN_BONUS"]
    if passed_pawn_score(board, 'black', state): score -= a*ENGINE_PARAMS["PASSED_PAWN_BONUS"] + b*ENGINE_PARAMS["PASSED_PAWN_BONUS"] + c*ENGINE_PARAMS["PASSED_PAWN_BONUS"]

    # Pawn chains returnerar bara True/False, så där behåller vi if-satsen
    if pawn_chain(board, 'white', state): score += a*ENGINE_PARAMS["pawn_chain_bonus"] + b*ENGINE_PARAMS["pawn_chain_bonus"] + c*ENGINE_PARAMS["pawn_chain_bonus"]
    if pawn_chain(board, 'black', state): score -= a*ENGINE_PARAMS["pawn_chain_bonus"] + b*ENGINE_PARAMS["pawn_chain_bonus"] + c*ENGINE_PARAMS["pawn_chain_bonus"]

    if rook_on_the_seventh_rank(board, 'white', state): score += a*ENGINE_PARAMS["rook_on_seventh_rank_bonus"] + b*ENGINE_PARAMS["rook_on_seventh_rank_bonus"] + c*ENGINE_PARAMS["rook_on_seventh_rank_bonus"]
    if rook_on_the_seventh_rank(board, 'black', state): score -= a*ENGINE_PARAMS["rook_on_seventh_rank_bonus"] + b*ENGINE_PARAMS["rook_on_seventh_rank_bonus"] + c*ENGINE_PARAMS["rook_on_seventh_rank_bonus"]

    # Här lägger vi till returvärdena direkt (de hanterar redan +/- beroende på färg)
    corner_w = endgame_push_king_enemy_to_corner(board, 'white', state)
    corner_b = endgame_push_king_enemy_to_corner(board, 'black', state)
    score += corner_w * ENGINE_PARAMS["enemy_king_corner_bonus"] * c
    score += corner_b * ENGINE_PARAMS["enemy_king_corner_bonus"] * c

    center_w = push_king_towards_center_penalty(board, 'white', state)
    center_b = push_king_towards_center_penalty(board, 'black', state)
    score += center_w * ENGINE_PARAMS["enemy_king_center_bonus"] * c
    score += center_b * ENGINE_PARAMS["enemy_king_center_bonus"] * c

    white_in_check = is_check(board, 'white', state)
    black_in_check = is_check(board, 'black', state)

    if len(state.white_pieces) + len(state.black_pieces) < 6:
        score += get_endgame_bonus(board, 'white', state)
        score -= get_endgame_bonus(board, 'black', state)
        score += king_confinement_and_distance_bonus(state)



    return score
# ==========================================
# 3. SÖKMOTORN
# ==========================================

def order_moves(moves, board, best_move=None):
    def move_score(move):
        if move == best_move:
            return 1000000

        (sr, sc), (er, ec) = move
        captured = board[er][ec]
        attacker = board[sr][sc]
        
        if captured == ' ' and attacker in ('♙', '♟') and sc != ec:
            captured = '♟' if attacker == '♙' else '♙'
            
        if captured != ' ':
            return 10 * abs(PIECE_VALUES.get(captured, 0)) - abs(PIECE_VALUES.get(attacker, 0))
        return 0

    moves.sort(key=move_score, reverse=True)
    return moves
def _is_capture_move(board, state, move):
    """True om draget slår en pjäs eller förvandlar en bonde."""
    (sr, sc), (er, ec) = move
    if board[er][ec] != ' ':
        return True
        
    piece = board[sr][sc]
    # En passant
    if piece in ('♙', '♟') and sc != ec and state.en_passant_target == (er, ec):
        return True
    # Bondeförvandling
    if piece in ('♙', '♟') and (er == 0 or er == 7):
        return True
        
    return False


def _terminal_score(current_color, depth=0):

    magnitude = 1_000_000 + depth
    return -magnitude if current_color == 'white' else magnitude

"""
Förbättrad quiescence_search + hjälpfunktioner.
Klistra in i rating.py (ersätt gamla quiescence_search, lägg till de nya
hjälpfunktionerna ovanför den).

FÖRÄNDRINGAR mot originalet:
1. Använder get_all_legal_moves(..., captures_only=True) när INTE i schack
   -> ingen onödig draggenerering/legalitetskoll av tysta drag vid varje löv.
2. Särbehandlar schack: om man står i schack finns inget "stand pat" - man
   MÅSTE svara. Då söks ALLA lagliga svar (inte bara slag), annars missar
   motorn taktik som kräver ett tyst kungdrag/blockering som schacksvar.
3. Delta pruning: hoppar över ett slag om ens bästa tänkbara utfall
   (stand_pat + slaget pjäsens värde + marginal) ändå inte kan nå alpha.
4. SEE (Static Exchange Evaluation): hoppar över slag som förlorar material
   efter en fullständig utbytesföljd på rutan (t.ex. Dxb2 där b2-bonden är
   väl försvarad) - sådana slag är i praktiken aldrig rätt drag i tyst läge
   och kostar annars mycket sökdjup för inget.
"""

import time

# -----------------------------------------------------------------------
# SEE (Static Exchange Evaluation)
# -----------------------------------------------------------------------

PIECE_ORDER_VALUE = {
    '♙': 1, '♟': 1,
    '♘': 3, '♞': 3,
    '♗': 3, '♝': 3,
    '♖': 5, '♜': 5,
    '♕': 9, '♛': 9,
    '♔': 10000, '♚': 10000,
    ' ': 0,
}


def _attackers_to_square(board, r, c, by_color):
    """Returnerar en sorterad lista (billigast först) med (ruta, pjäsvärde)
    för alla pjäser av by_color som angriper (r, c). Används av SEE för att
    simulera en hel utbytesföljd utan att behöva göra riktiga drag/ångra."""
    attackers = []

    if by_color == 'white':
        own_set = frozenset('♙♘♗♖♕♔')
        pawn_dr, pawn_piece = 1, '♙'
    else:
        own_set = frozenset('♟♞♝♜♛♚')
        pawn_dr, pawn_piece = -1, '♟'

    # Bönder
    pr = r + pawn_dr
    for dc in (-1, 1):
        pc = c + dc
        if 0 <= pr < 8 and 0 <= pc < 8 and board[pr][pc] == pawn_piece:
            attackers.append(((pr, pc), PIECE_ORDER_VALUE[pawn_piece]))

    # Springare
    knight = '♘' if by_color == 'white' else '♞'
    for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == knight:
            attackers.append(((nr, nc), PIECE_ORDER_VALUE[knight]))

    # Kung
    king = '♔' if by_color == 'white' else '♚'
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            kr, kc = r + dr, c + dc
            if 0 <= kr < 8 and 0 <= kc < 8 and board[kr][kc] == king:
                attackers.append(((kr, kc), PIECE_ORDER_VALUE[king]))

    # Torn/Dam (raka linjer)
    rook_like = {'♖', '♕'} if by_color == 'white' else {'♜', '♛'}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, rc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= rc < 8:
            p = board[rr][rc]
            if p != ' ':
                if p in rook_like:
                    attackers.append(((rr, rc), PIECE_ORDER_VALUE[p]))
                break
            rr += dr
            rc += dc

    # Löpare/Dam (diagonaler)
    bishop_like = {'♗', '♕'} if by_color == 'white' else {'♝', '♛'}
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr, rc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= rc < 8:
            p = board[rr][rc]
            if p != ' ':
                if p in bishop_like:
                    attackers.append(((rr, rc), PIECE_ORDER_VALUE[p]))
                break
            rr += dr
            rc += dc

    attackers.sort(key=lambda x: x[1])
    return attackers


def static_exchange_eval(board, move):
    """Simulerar en fullständig utbytesföljd på målrutan UTAN att röra det
    riktiga brädet/state (allt sker på en kopia), och returnerar nettovinsten
    i pjäspoäng (positivt = lönsamt slag för den som drar).

    Standardalgoritm: båda sidor tar tillbaka med sin billigaste tillgängliga
    pjäs tills ingen vill/kan fortsätta (negamax över utbytesserien)."""
    (sr, sc), (er, ec) = move
    attacker_piece = board[sr][sc]
    target_piece = board[er][ec]
    if target_piece == ' ':
        return 0  # tyst drag (en passant hanteras separat i _is_capture_move)

    attacker_color = 'white' if attacker_piece in '♙♘♗♖♕♔' else 'black'
    defender_color = 'black' if attacker_color == 'white' else 'white'

    board_copy = [row[:] for row in board]
    board_copy[er][ec] = attacker_piece
    board_copy[sr][sc] = ' '

    gain = [PIECE_ORDER_VALUE[target_piece]]
    current_attacker_value = PIECE_ORDER_VALUE[attacker_piece]
    side = defender_color

    while True:
        attackers = _attackers_to_square(board_copy, er, ec, side)
        if not attackers:
            break
        (fr, fc), val = attackers[0]
        gain.append(current_attacker_value - gain[-1])
        board_copy[er][ec] = board_copy[fr][fc]
        board_copy[fr][fc] = ' '
        current_attacker_value = val
        side = 'black' if side == 'white' else 'white'

    # Vik ihop utbytesserien bakifrån (negamax): varje sida väljer att
    # AVBRYTA utbytet (dvs. inte återta) om det skulle bli en förlustaffär,
    # så varje steg tar det bästa av "fortsätt utbytet" vs "sluta här".
    for i in range(len(gain) - 2, -1, -1):
        gain[i] = min(gain[i], -gain[i + 1])
    return gain[0]

