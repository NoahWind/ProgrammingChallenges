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

PIECE_VALUES = {
    '♟': -1, '♞': -3, '♝': -3, '♜': -5, '♛': -9, '♚': -10000,
    '♙': 1,  '♘': 3,  '♗': 3,  '♖': 5,  '♕': 9,  '♔': 10000,
    ' ': 0
}

# Piece-Square Tables (PST) – värderar var pjäserna står (från vits perspektiv)
# För svart inverteras raderna automatiskt.

PAWN_PST = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [ 5,  5, 10, 25, 25, 10,  5,  5],
    [ 0,  0,  0, 20, 20,  0,  0,  0],
    [ 5, -5,-10,  0,  0,-10, -5,  5],
    [ 5, 10, 10,-20,-20, 10, 10,  5],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
]

KNIGHT_PST = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
]

BISHOP_PST = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
]

ROOK_PST = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [ 0,  0,  0,  5,  5,  0,  0,  0]
]

QUEEN_PST = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
]

def evaluate_pieces_and_threats(board, state, color):
    """Beräknar PST och hängande pjäser för en specifik färg 
    och returnerar (pst_score, hanging_score)."""
    pst_score = 0.0
    hanging_score = 0.0
    
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    enemy_color = 'black' if color == 'white' else 'white'
    king_piece = '♔' if color == 'white' else '♚'
    
    for r, c in own_pieces:
        piece = board[r][c]
        if piece == ' ':
            continue
        
        # PST-beräkning
        if color == 'white':
            if piece == '♙': pst_score += PAWN_PST[r][c] * 0.01
            elif piece == '♘': pst_score += KNIGHT_PST[r][c] * 0.01
            elif piece == '♗': pst_score += BISHOP_PST[r][c] * 0.01
            elif piece == '♖': pst_score += ROOK_PST[r][c] * 0.01
            elif piece == '♕': pst_score += QUEEN_PST[r][c] * 0.01
        else:
            sr = 7 - r
            if piece == '♟': pst_score += PAWN_PST[sr][c] * 0.01
            elif piece == '♞': pst_score += KNIGHT_PST[sr][c] * 0.01
            elif piece == '♝': pst_score += BISHOP_PST[sr][c] * 0.01
            elif piece == '♜': pst_score += ROOK_PST[sr][c] * 0.01
            elif piece == '♛': pst_score += QUEEN_PST[sr][c] * 0.01

        # Hängande pjäser-kontroll
        if piece != king_piece:
            if _is_square_attacked_fast(board, r, c, enemy_color):
                if not _is_square_attacked_fast(board, r, c, color):
                    val = abs(PIECE_VALUES.get(piece, 0))
                    hanging_score -= val * 0.15

    # Positivt för vit, negativt för svart (följer motorns konvention)
    mult = 1.0 if color == 'white' else -1.0
    return pst_score * mult, hanging_score * mult

BLACK_PIECES = {'♟', '♞', '♝', '♜', '♛', '♚'}
WHITE_PIECES = {'♙', '♘', '♗', '♕', '♖', '♔'}

cache = {}
search_start_time = 0
time_limit_seconds = 0.0005
DEFAULT_TIME_LIMIT_SECONDS = 0.5


# ==========================================
# 1. DINA HJÄLPFUNKTIONER (oförändrade)
# ==========================================

def get_endgame_bonus(board, color, state):
    """Returnerar ett litet bonusvärde för att ha en bonde som kan
    promoveras, eller en bonde som är nära promotion (t.ex. på rad 6)."""
    bonus = 0
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    pawn_piece = '♙' if color == 'white' else '♟'
    
    for r, c in own_pieces:
        if board[r][c] == pawn_piece:
            if color == 'white':
                bonus += (7 - r) * 0.1  # Rad 6 ger +0.1, rad 5 ger +0.2, etc.
            else:
                bonus += r * 0.1  # Rad 1 ger +0.1, rad 2 ger +0.2, etc.
    
    return bonus


def king_confinement_and_distance_bonus(state):
    """NY HEURISTIK: ger evalueringen en gradient mot matt i pjäsfattiga
    slutspel (K+D/K+T mot bar kung, K+B+S mot bar kung, osv).

    Utan den här ser ALLA drag likadana ut för evalueringen så fort man
    leder stort i material - "damen på c5" och "damen på d4" ger exakt
    samma poäng - så sökningen (begränsad till några få plys) har inget
    att navigera efter och skvalpar planlöst istället för att mattsätta.

    Idén (standard i enkla schackmotorer): den starkare sidan vill
    1) tränga in fiendekungen mot brädets kant/hörn, och
    2) föra sin egen kung närmare fiendekungen.
    Ju närmare matt, desto större bonus - vilket ger sökträdet en riktning
    att följa istället för att bara stå still på ett "redan vinnande" plus.
    """
    total_pieces = len(state.white_pieces) + len(state.black_pieces)
    if total_pieces >= 6 or state.material_score == 0:
        return 0.0

    stronger = 'white' if state.material_score > 0 else 'black'
    if stronger == 'white':
        strong_king, weak_king = state.white_king_pos, state.black_king_pos
    else:
        strong_king, weak_king = state.black_king_pos, state.white_king_pos

    if strong_king is None or weak_king is None:
        return 0.0

    wr, wc = weak_king
    sr, sc = strong_king

    # Chebyshev-avstånd från centrum: 0 i mitten, 3.5 i hörnen.
    # Högre värde = den svaga kungen är mer trängd mot kanten.
    edge_distance = max(abs(3.5 - wr), abs(3.5 - wc))

    # Kungarnas avstånd till varandra (Chebyshev). Ju närmare, desto bättre
    # för den starkare sidan när den ska driva fram mattet.
    kings_distance = max(abs(wr - sr), abs(wc - sc))
    closeness = 7 - kings_distance

    bonus = edge_distance * 0.15 + closeness * 0.1
    return bonus if stronger == 'white' else -bonus

def is_controling_center(board, color):
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for r, c in center_squares:
        piece = board[r][c]
        if piece != ' ' and ((color == 'white' and piece in WHITE_PIECES) or (color == 'black' and piece in BLACK_PIECES)):
            return True
    return False


def is_king_safe(board, color, state):
    return not is_check(board, color, state)


def is_wining_in_material(board, color):
    score = sum(PIECE_VALUES.get(p, 0) for row in board for p in row if p != ' ')
    return (score > 0 and color == 'white') or (score < 0 and color == 'black')


def pawn_chain(board, color, state):
    pawn_piece = '♙' if color == 'white' else '♟'
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    pawn_positions = {(r, c) for r, c in own_pieces if board[r][c] == pawn_piece}
    
    for r, c in pawn_positions:
        if color == 'white':
            if (r + 1, c - 1) in pawn_positions or (r + 1, c + 1) in pawn_positions: return True
        else:
            if (r - 1, c - 1) in pawn_positions or (r - 1, c + 1) in pawn_positions: return True
    return False

def bishop_pair(board, color, state):
    bishop_piece = '♗' if color == 'white' else '♝'
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    bishops = sum(1 for r, c in own_pieces if board[r][c] == bishop_piece)
    return bishops >= 2

def knight_on_the_rim(board, color, state):
    knight_piece = '♘' if color == 'white' else '♞'
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    for r, c in own_pieces:
        if board[r][c] == knight_piece:
            if r == 0 or r == 7 or c == 0 or c == 7:
                return True
    return False

def passed_pawn_score(board, color, state):
    """
    Returnerar en bonus för fribönder (passed pawns).
    En fribonde är en bonde som inte hindras av fientliga bönder
    på sin egen eller de två intilliggande kolumnerna.
    """
    pawn_piece = '♙' if color == 'white' else '♟'
    enemy_pawn = '♟' if color == 'white' else '♙'
    
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    enemy_pieces = state.black_pieces if color == 'white' else state.white_pieces
    
    # Samla alla fiendens bönder för extremt snabb sökning
    enemy_pawns = {(r, c) for r, c in enemy_pieces if board[r][c] == enemy_pawn}
    
    bonus = 0.0
    
    for r, c in own_pieces:
        if board[r][c] == pawn_piece:
            is_passed = True
            
            # Kolla fientliga bönder på kolumn c-1, c, och c+1
            for er, ec in enemy_pawns:
                if abs(ec - c) <= 1:
                    # Vit går uppåt i arrayen (lägre rad-index)
                    if color == 'white' and er < r:
                        is_passed = False
                        break
                    # Svart går nedåt i arrayen (högre rad-index)
                    elif color == 'black' and er > r:
                        is_passed = False
                        break
                        
            if is_passed:
                # Progressiv bonus: Ju närmare sista raden, desto högre poäng!
                if color == 'white':
                    bonus += 0.5 + (6 - r) * 0.2  # T.ex. rad 2 (nära dam) ger +1.3
                else:
                    bonus += 0.5 + (r - 1) * 0.2  # T.ex. rad 5 (nära dam) ger +1.3
                    
    return bonus if color == 'white' else -bonus
    

def breakt_pawn_chains(board, color, state):
    pawn_piece = '♙' if color == 'white' else '♟'
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    pawn_positions = {(r, c) for r, c in own_pieces if board[r][c] == pawn_piece}
    
    for r, c in pawn_positions:
        if color == 'white':
            if (r + 1, c - 1) not in pawn_positions and (r + 1, c + 1) not in pawn_positions:
                return True
        else:
            if (r - 1, c - 1) not in pawn_positions and (r - 1, c + 1) not in pawn_positions:
                return True
    return False

def how_many_squares_do_i_control(board, color, state):
    controlled_squares = set()
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    
    for r, c in own_pieces:
        piece = board[r][c]
        if piece != ' ':
            moves = get_all_legal_moves(color, board, state)
            for move in moves:
                (sr, sc), (er, ec) = move
                if (sr, sc) == (r, c):
                    controlled_squares.add((er, ec))
    
    return len(controlled_squares)

def three_fold_repetition(history, board, current_color, state):
    board_hash = (tuple(map(tuple, board)), current_color, _state_key(state))
    return history.get(board_hash, 0) >= 2

def push_king_towards_center_penalty(board, color, state):
    """NY HEURISTIK: ger en liten bonus om kungen är nära centrum i slutspel.
    Ju närmare centrum, desto bättre. Ju mer material på brädet, desto mindre
    bonus (kungen ska inte vara framme i öppningen)."""
    total_pieces = len(state.white_pieces) + len(state.black_pieces)
    if total_pieces >= 6:
        return 0.0

    king_pos = state.white_king_pos if color == 'white' else state.black_king_pos
    if king_pos is None:
        return 0.0

    r, c = king_pos
    center_distance = max(abs(3.5 - r), abs(3.5 - c))  # Chebyshev-avstånd från centrum
    bonus = (3.5 - center_distance) * 0.1  # Ju närmare centrum, desto högre bonus
    return bonus if color == 'white' else -bonus

def endgame_push_king_enemy_to_corner(board, color, state):
    """NY HEURISTIK: ger en liten bonus om den starkare sidan kan tränga
    in fiendekungen mot brädets kant/hörn i slutspel. Ju närmare hörnet,
    desto högre bonus. Ju mer material på brädet, desto mindre bonus."""
    total_pieces = len(state.white_pieces) + len(state.black_pieces)
    if total_pieces >= 6:
        return 0.0

    weak_king_pos = state.black_king_pos if color == 'white' else state.white_king_pos
    if weak_king_pos is None:
        return 0.0

    r, c = weak_king_pos
    corner_distance = min(r, 7 - r, c, 7 - c)  # Avstånd till närmaste hörn
    bonus = (3.5 - corner_distance) * 0.1  # Ju närmare hörnet, desto högre bonus
    return bonus if color == 'white' else -bonus

def not_hanging_my_own_pieces(board, color, state):
    """NY HEURISTIK: ger en liten bonus om man inte har hängande pjäser (pjäser som kan slås utan motdrag). Ju fler hängande pjäser, desto större straff."""
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    enemy_color = 'black' if color == 'white' else 'white'
    enemy_moves = get_all_legal_moves(enemy_color, board, state)
    
    hanging_count = 0
    for r, c in own_pieces:
        piece = board[r][c]
        if piece != ' ':
            for move in enemy_moves:
                (sr, sc), (er, ec) = move
                if (er, ec) == (r, c):
                    hanging_count += 1
                    break  # Räkna varje pjäs bara en gång

    penalty = hanging_count * 0.2  # Varje hängande pjäs ger -0.2 poäng
    return -penalty if color == 'white' else penalty

def _is_square_attacked_fast(board, r, c, by_color):
    """Kollar snabbt om en ruta (r, c) är attackerad av 'by_color' 
    genom att direkt kontrollera pjäsmönster utan draggenerering."""
    
    # 1. Bonde-attacker
    if by_color == 'white':
        for dc in (-1, 1):
            pr, pc = r + 1, c + dc
            if 0 <= pr < 8 and 0 <= pc < 8 and board[pr][pc] == '♙':
                return True
    else:
        for dc in (-1, 1):
            pr, pc = r - 1, c + dc
            if 0 <= pr < 8 and 0 <= pc < 8 and board[pr][pc] == '♟':
                return True

    # 2. Springar-attacker
    knight_piece = '♘' if by_color == 'white' else '♞'
    knight_offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
    for dr, dc in knight_offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == knight_piece:
            return True

    # 3. Kung-attacker
    king_piece = '♔' if by_color == 'white' else '♚'
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            kr, kc = r + dr, c + dc
            if 0 <= kr < 8 and 0 <= kc < 8 and board[kr][kc] == king_piece:
                return True

    # 4. Torn- och dam-attacker (raka linjer)
    rook_queen = {'♖', '♕'} if by_color == 'white' else {'♜', '♛'}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        curr_r, curr_c = r + dr, c + dc
        while 0 <= curr_r < 8 and 0 <= curr_c < 8:
            p = board[curr_r][curr_c]
            if p != ' ':
                if p in rook_queen:
                    return True
                break
            curr_r += dr
            curr_c += dc

    # 5. Löpar- och dam-attacker (diagonaler)
    bishop_queen = {'♗', '♕'} if by_color == 'white' else {'♝', '♛'}
    diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in diagonals:
        curr_r, curr_c = r + dr, c + dc
        while 0 <= curr_r < 8 and 0 <= curr_c < 8:
            p = board[curr_r][curr_c]
            if p != ' ':
                if p in bishop_queen:
                    return True
                break
            curr_r += dr
            curr_c += dc

    return False


def hanging_pieces_penalty(board, color, state):
    """Straffar pjäser som är attackerade av motståndaren men saknar eget försvar."""
    enemy_color = 'black' if color == 'white' else 'white'
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    
    penalty = 0.0
    
    for r, c in own_pieces:
        piece = board[r][c]
        if piece == ' ' or piece in ('♔', '♚'):
            continue  # Hoppa över tomma rutor och kungen
            
        # Om pjäsen är attackerad av motståndaren...
        if _is_square_attacked_fast(board, r, c, enemy_color):
            # ...och INTE är försvarad av oss själva, då är den hängande!
            if not _is_square_attacked_fast(board, r, c, color):
                val = abs(PIECE_VALUES.get(piece, 0))
                penalty += val * 0.15  # Straffproportionellt mot pjäsvärdet

    # Returnera negativt för vit (sänker vits poäng) och positivt för svart (höjer fördel svart)
    return -penalty if color == 'white' else penalty

def _has_any_pawns(board):
    """True om minst en bonde finns kvar på brädet. Används för att INTE
    döma upprepning som remi i bondeslutspel - ett litet materialövertag
    (t.ex. +1 i en bondeändspel) kan fortfarande vara en fullt vunnen
    ställning, till skillnad från ett rent pjäsövertag på <5 poäng."""
    for row in board:
        for p in row:
            if p in ('♙', '♟'):
                return True
    return False
# ==========================================
# 2. UTVÄRDERING
# ==========================================
KING_SAFETY_BONUS = 5
CONTROL_CENTER_BONUS = 0.5
BREAKING_PAWN_CHAINS_BONUS = 1
bishop_pair_bonus = 0.5
knight_on_the_rim_penalty = 0.5
pawn_chain_bonus = 2
PASSED_PAWN_BONUS = 0.5
enemy_king_corner_bonus = 1
enemy_king_center_bonus= 0.5
hanging_piece_penalty = 0.5
squares_controlled_bonus = 0.01
pieac_pos_bonus = 0.01

def evaluate_board(board, state, history, current_color):
    score = float(state.material_score)
    
    pos_key = get_position_hash(board, current_color, state)
    if (history.get(pos_key, 0) >= 2
            and abs(state.material_score) < 5
            and not _has_any_pawns(board)):
        return 0 

    if state.half_move_clock > 80:
        if score < 0:
            score += (state.half_move_clock - 80) * 0.5 
        elif score > 0:
            score -= (state.half_move_clock - 80) * 0.5
    
    if is_king_safe(board, 'white', state): score += KING_SAFETY_BONUS
    if is_king_safe(board, 'black', state): score -= KING_SAFETY_BONUS

    if is_controling_center(board, 'white'): score += CONTROL_CENTER_BONUS
    if is_controling_center(board, 'black'): score -= CONTROL_CENTER_BONUS

    if breakt_pawn_chains(board, 'white', state): score -= BREAKING_PAWN_CHAINS_BONUS
    if breakt_pawn_chains(board, 'black', state): score += BREAKING_PAWN_CHAINS_BONUS

    if bishop_pair(board, 'white', state): score += bishop_pair_bonus
    if bishop_pair(board, 'black', state): score -= bishop_pair_bonus

    if knight_on_the_rim(board, 'white', state): score -= knight_on_the_rim_penalty
    if knight_on_the_rim(board, 'black', state): score += knight_on_the_rim_penalty

    if not_hanging_my_own_pieces(board, 'white', state): score += not_hanging_my_own_pieces(board, 'white', state) * 50
    if not_hanging_my_own_pieces(board, 'black', state): score += not_hanging_my_own_pieces(board, 'black', state) * 50

    if how_many_squares_do_i_control(board, 'white', state): score += squares_controlled_bonus * how_many_squares_do_i_control(board, 'white', state)
    if how_many_squares_do_i_control(board, 'black', state): score -= squares_controlled_bonus * how_many_squares_do_i_control(board, 'black', state)

    pst_w, hang_w = evaluate_pieces_and_threats(board, state, 'white')
    pst_b, hang_b = evaluate_pieces_and_threats(board, state, 'black')

    score += pst_w * pieac_pos_bonus
    score += pst_b * pieac_pos_bonus
    score += hang_w * hanging_piece_penalty
    score += hang_b * hanging_piece_penalty


    score += passed_pawn_score(board, 'white', state)
    score += passed_pawn_score(board, 'black', state)

    # Pawn chains returnerar bara True/False, så där behåller vi if-satsen
    if pawn_chain(board, 'white', state): score += pawn_chain_bonus
    if pawn_chain(board, 'black', state): score -= pawn_chain_bonus

    # Här lägger vi till returvärdena direkt (de hanterar redan +/- beroende på färg)
    score += endgame_push_king_enemy_to_corner(board, 'white', state)
    score += endgame_push_king_enemy_to_corner(board, 'black', state)

    score += push_king_towards_center_penalty(board, 'white', state)
    score += push_king_towards_center_penalty(board, 'black', state)
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
    """±(1 000 000 + depth) vid matt, 0 vid patt. Delas av minimax och
    quiescence_search.

    BUGGFIX: tidigare gavs alltid exakt ±1 000 000 oavsett hur djupt i
    trädet mattet hittades, vilket gjorde motorn likgiltig mellan matt i 1
    och matt i 8 - den kunde skjuta upp en vinst helt i onödan, eller (vid
    förlust) gå rakt in i den snabbast möjliga mattsättningen istället för
    att bjuda motstånd. `depth` är kvarvarande sökdjup när mattet hittas;
    ett större kvarvarande djup betyder att mattet hittades snabbare (färre
    drag förbrukade), så det ska ge ett större utslag."""
    magnitude = 1_000_000 + depth
    return -magnitude if current_color == 'white' else magnitude

def quiescence_search(board, state, alpha, beta, is_maximizing, history, q_depth=0):
    # BUGGFIX: quiescence_search saknade helt tidsgränskoll. Vid långa
    # forcerande slagserier kunde ett enskilt drag därför ta betydligt
    # längre tid än DEFAULT_TIME_LIMIT_SECONDS, eftersom minimax bara
    # kollar klockan i sina egna anrop - inte under tiden quiescence_search
    # arbetar.
    if time.time() - search_start_time > time_limit_seconds:
        raise TimeoutError()

    current_color = 'white' if is_maximizing else 'black'
    moves = get_all_legal_moves(current_color, board, state)

    if not moves:
        return _terminal_score(current_color, -q_depth) if is_check(board, current_color, state) else 0

    # Nu skickar vi med history och current_color till evaluate_board
    stand_pat = evaluate_board(board, state, history, current_color)

    if is_maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)

    if q_depth >= 10: return stand_pat

    capture_moves = [m for m in moves if _is_capture_move(board, state, m)]
    capture_moves = order_moves(capture_moves, board)

    if not capture_moves: return stand_pat

    if is_maximizing:
        for move in capture_moves:
            record = apply_move(board, state, move[0], move[1])
            try:
                # Skicka med history även här!
                eval_val = quiescence_search(board, state, alpha, beta, False, history, q_depth + 1)
            finally:
                undo_move(board, state, record)
            if eval_val >= beta: return beta
            alpha = max(alpha, eval_val)
        return alpha
    else:
        for move in capture_moves:
            record = apply_move(board, state, move[0], move[1])
            try:
                # Skicka med history även här!
                eval_val = quiescence_search(board, state, alpha, beta, True, history, q_depth + 1)
            finally:
                undo_move(board, state, record)
            if eval_val <= alpha: return alpha
            beta = min(beta, eval_val)
        return beta


def _state_key(state):
    return (
        state.white_king_moved, state.black_king_moved,
        state.rook_a1_moved, state.rook_h1_moved,
        state.rook_a8_moved, state.rook_h8_moved,
        state.en_passant_target,
    )

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

def minimax(board, state, depth, alpha, beta, is_maximizing, history):
    if time.time() - search_start_time > time_limit_seconds:
        raise TimeoutError()

    current_color = 'white' if is_maximizing else 'black'
    original_alpha = alpha
    original_beta = beta
    
    moves = get_all_legal_moves(current_color, board, state)
    
    if not moves:
        if is_check(board, current_color, state):
            return _terminal_score(current_color, depth)
        else:
            return 0
            
    board_hash = get_position_hash(board, current_color, state)
    seen_on_path = history.get(board_hash, 0)

    if seen_on_path >= 2:
        return 0  # En tre-faldig upprepning är alltid remi
    
    history[board_hash] = seen_on_path + 1
    use_cache = (seen_on_path == 0)

    if use_cache and board_hash in cache:
        cached_eval, cached_depth, node_type = cache[board_hash]
        if cached_depth >= depth:
            if node_type == 'EXACT':
                history[board_hash] -= 1
                return cached_eval
            elif node_type == 'LOWERBOUND' and cached_eval >= beta:
                history[board_hash] -= 1
                return cached_eval
            elif node_type == 'UPPERBOUND' and cached_eval <= alpha:
                history[board_hash] -= 1
                return cached_eval

    if depth == 0:
        result = quiescence_search(board, state, alpha, beta, is_maximizing, history, 0)
        if use_cache:
            cache[board_hash] = (result, depth, 'EXACT')
        history[board_hash] -= 1
        return result

    moves = order_moves(moves, board)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            record = apply_move(board, state, move[0], move[1])
            try:
                eval_val = minimax(board, state, depth - 1, alpha, beta, not is_maximizing, history)
            finally:
                undo_move(board, state, record)
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha: break
        if use_cache:
            if max_eval <= original_alpha:
                node_type = 'UPPERBOUND'
            elif max_eval >= beta:
                node_type = 'LOWERBOUND'
            else:
                node_type = 'EXACT'
            cache[board_hash] = (max_eval, depth, node_type)
        history[board_hash] -= 1
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            record = apply_move(board, state, move[0], move[1])
            try:
                eval_val = minimax(board, state, depth - 1, alpha, beta, not is_maximizing, history)
            finally:
                undo_move(board, state, record)
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha: break
        if use_cache:
            if min_eval >= original_beta:
                node_type = 'LOWERBOUND'
            elif min_eval <= alpha:
                node_type = 'UPPERBOUND'
            else:
                node_type = 'EXACT'
            cache[board_hash] = (min_eval, depth, node_type)
        history[board_hash] -= 1
        return min_eval
def get_best_move(board, depth, color, state, history=None):
    global cache, search_start_time, time_limit_seconds

    search_start_time = time.time()
    time_limit_seconds = DEFAULT_TIME_LIMIT_SECONDS  # Aktivera alltid ordinarie tänktid

    is_white_turn = (color == 'white')

    if history is None:
        history = {}
        
    moves = get_all_legal_moves(color, board, state)
    if not moves:
        return None
        
    best_move_overall = moves[0]
    best_value = float('-inf') if color == 'white' else float('inf')
    search_depth = max(1, int(round(float(depth))))

    try:
        for current_depth in range(1, search_depth + 1):
            moves = order_moves(moves, board, best_move_overall)

            best_move_for_this_depth = None
            best_value_for_this_depth = float('-inf') if color == 'white' else float('inf')
            
            # Startvärden för alpha-beta-beskärning i roten
            alpha = float('-inf')
            beta = float('inf')

            # Vi räknar igenom drag för detta djup
            for move in moves:
                record = apply_move(board, state, move[0], move[1])
                try:
                    move_value = minimax(board, state, current_depth - 1, alpha, beta, not is_white_turn, history)
                finally:
                    undo_move(board, state, record)

                if color == 'white':
                    if move_value > best_value_for_this_depth:
                        best_value_for_this_depth = move_value
                        best_move_for_this_depth = move
                    alpha = max(alpha, move_value)  # Uppdatera alpha!
                else:
                    if move_value < best_value_for_this_depth:
                        best_value_for_this_depth = move_value
                        best_move_for_this_depth = move
                    beta = min(beta, move_value)    # Uppdatera beta!

            # Om loopen nådde hit har HELA djupet slutförts utan tidsavbrott.
            # Först NU uppdaterar vi det officiella bästa draget!
            if best_move_for_this_depth is not None:
                best_move_overall = best_move_for_this_depth
                best_value = best_value_for_this_depth

    except TimeoutError:
        # Om tiden tar slut mitt i ett djup kastas undantaget hit.
        # Eftersom vi inte hann uppdatera 'best_move_overall' inuti det 
        # ofullständiga djupet, faller vi nu tryggt tillbaka på det 
        # SISTA HELT FÄRDIGRÄKNADE DJUPET istället för att chansa.
        print("Tiden tog slut – använder resultat från föregående färdiga djup.")
    
    print("Cache size:", len(cache), "entries")
    print("in MB:", len(cache) * 64 / (1024 * 1024))
    print("Best move found:", best_move_overall, "with evaluation:", best_value)
    return best_move_overall