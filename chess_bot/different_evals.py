from chess_rules import get_all_legal_moves, get_location_of_all_pieces, is_check, apply_move, undo_move, is_stalemate, get_position_hash


PIECE_VALUES = {
    '♟': -1, '♞': -3, '♝': -3, '♜': -5, '♛': -9, '♚': -10000,
    '♙': 1,  '♘': 3,  '♗': 3,  '♖': 5,  '♕': 9,  '♔': 10000,
    ' ': 0
}

def _state_key(state):
    return (
        state.white_king_moved, state.black_king_moved,
        state.rook_a1_moved, state.rook_h1_moved,
        state.rook_a8_moved, state.rook_h8_moved,
        state.en_passant_target,
    )


# Piece-Square Tables (PST) – värderar var pjäserna står (från vits perspektiv)
# För svart inverteras raderna automatiskt.

# King Piece-Square Table för öppning och mittspel
# Straffar kungen hårt i centrum, belönar kanten och rockad-rutorna (g1/h1/b1/c1)
KING_MG_PST = [
    [-50, -40, -40, -50, -50, -40, -40, -50],
    [-50, -40, -40, -50, -50, -40, -40, -50],
    [-50, -40, -40, -50, -50, -40, -40, -50],
    [-50, -40, -40, -50, -50, -40, -40, -50],
    [-30, -30, -30, -40, -40, -30, -30, -30],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 30,  40,  10, -20, -20,  10,  40,  30]
]

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


BLACK_PIECES = {'♟', '♞', '♝', '♜', '♛', '♚'}
WHITE_PIECES = {'♙', '♘', '♗', '♕', '♖', '♔'}

cache = {}
search_start_time = 0
time_limit_seconds = 0.000005


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

def build_attack_map(board, by_color, state):
    """Bygger en set av alla rutor som attackerats av by_color. O(pjäser) istället för O(rutor)."""
    attacked = set()
    own_pieces = state.white_pieces if by_color == 'white' else state.black_pieces
    
    pawn_dr = -1 if by_color == 'white' else 1
    pawn_piece = '♙' if by_color == 'white' else '♟'
    knight_piece = '♘' if by_color == 'white' else '♞'
    king_piece = '♔' if by_color == 'white' else '♚'
    rook_like = {'♖', '♕'} if by_color == 'white' else {'♜', '♛'}
    bishop_like = {'♗', '♕'} if by_color == 'white' else {'♝', '♛'}

    for r, c in own_pieces:
        piece = board[r][c]
        
        # Bönder
        if piece == pawn_piece:
            pr = r + pawn_dr
            if 0 <= pr < 8:
                if c - 1 >= 0: attacked.add((pr, c - 1))
                if c + 1 < 8: attacked.add((pr, c + 1))
                
        # Springare
        elif piece == knight_piece:
            for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    attacked.add((nr, nc))
                    
        # Kung
        elif piece == king_piece:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        attacked.add((nr, nc))
                        
        # Glidande pjäser (Torn, Löpare, Dam)
        else:
            dirs = []
            if piece in rook_like:
                dirs.extend(((-1, 0), (1, 0), (0, -1), (0, 1)))
            if piece in bishop_like:
                dirs.extend(((-1, -1), (-1, 1), (1, -1), (1, 1)))
                
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    attacked.add((nr, nc))
                    if board[nr][nc] != ' ':
                        break
                    nr += dr
                    nc += dc
                    
    return attacked


def is_king_safe(board, color, state, enemy_attacks):
    """Kollar kungsäkerhet blixtsnabbt med hjälp av den färdiga hot-kartan."""
    king_pos = state.white_king_pos if color == 'white' else state.black_king_pos
    if king_pos is None:
        return False

    kr, kc = king_pos

    # 1. Om kungen står i schack (dess ruta finns i hot-kartan)
    if king_pos in enemy_attacks:
        return False

    # 2. Kolla bondeskölden framför kungen
    shield_row = kr - 1 if color == 'white' else kr + 1
    pawn_piece = '♙' if color == 'white' else '♟'
    
    pawn_shield_count = 0
    if 0 <= shield_row < 8:
        for c_offset in (-1, 0, 1):
            nc = kc + c_offset
            if 0 <= nc < 8 and board[shield_row][nc] == pawn_piece:
                pawn_shield_count += 1

    # 3. Räkna rutor i 3x3-zonen som är attackerade av fienden
    threatened_squares = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = kr + dr, kc + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if (nr, nc) in enemy_attacks:
                    threatened_squares += 1

    return (pawn_shield_count >= 2) and (threatened_squares <= 1)


def evaluate_pieces_and_threats(board, state, color, own_attacks, enemy_attacks):
    """Utvärderar pjäser, hängande hot samt pjäser inställda i bondeslag."""
    pst_score = 0.0
    hanging_score = 0.0
    
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    king_piece = '♔' if color == 'white' else '♚'
    enemy_color = 'black' if color == 'white' else 'white'
    enemy_pawn = '♟' if color == 'white' else '♙'
    pawn_dr = 1 if color == 'white' else -1  # Riktningen fiendebonden anfaller ifrån
    
    for r, c in own_pieces:
        piece = board[r][c]
        if piece == ' ':
            continue
        
        # 1. PST-beräkning (Piece-Square Tables)
        if color == 'white':
            if piece == '♙': pst_score += PAWN_PST[r][c] * 0.01
            elif piece == '♘': pst_score += KNIGHT_PST[r][c] * 0.01
            elif piece == '♗': pst_score += BISHOP_PST[r][c] * 0.01
            elif piece == '♖': pst_score += ROOK_PST[r][c] * 0.01
            elif piece == '♕': pst_score += QUEEN_PST[r][c] * 0.01
            elif piece == '♔': pst_score += KING_MG_PST[r][c] * 0.01
        else:
            sr = 7 - r
            if piece == '♟': pst_score += PAWN_PST[sr][c] * 0.01
            elif piece == '♞': pst_score += KNIGHT_PST[sr][c] * 0.01
            elif piece == '♝': pst_score += BISHOP_PST[sr][c] * 0.01
            elif piece == '♜': pst_score += ROOK_PST[sr][c] * 0.01
            elif piece == '♛': pst_score += QUEEN_PST[sr][c] * 0.01
            elif piece == '♚': pst_score += KING_MG_PST[sr][c] * 0.01

        # 2. Hot- och säkerhetskontroller (hoppa över kungen)
        if piece != king_piece:
            val = abs(PIECE_VALUES.get(piece, 0))

            # KOLL A: Står pjäsen i ett fientligt bondeslag?
            # Om pjäsen är mer värd än en bonde (D, T, L, S) straffas den hårt
            # även om den råkar ha ett eget försvar bakom sig!
            is_attacked_by_pawn = False
            pr = r + pawn_dr
            if 0 <= pr < 8:
                if (c - 1 >= 0 and board[pr][c - 1] == enemy_pawn) or \
                   (c + 1 < 8 and board[pr][c + 1] == enemy_pawn):
                    is_attacked_by_pawn = True

            if is_attacked_by_pawn and val > 1:
                # Straffa förlusten av pjäsen minus bondens värde (val - 1)
                hanging_score -= (val - 1)
            
            # KOLL B: Helt oskyddad/hängande pjäs
            elif (r, c) in enemy_attacks and (r, c) not in own_attacks:
                hanging_score -= val

    mult = 1.0 if color == 'white' else -1.0
    return pst_score * mult, hanging_score * mult

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

def how_many_squares_do_i_control(board, color, state,moves):
    # get_all_legal_moves beror bara på board/state/color - inte på vilken
    # egen pjäs vi råkar loopa över just nu, så den ska bara anropas EN gång
    # totalt, inte en gång per egen pjäs (var tidigare upp till ~16x dyrare
    # än nödvändigt, eftersom hela draglistan räknades om för varje pjäs
    # bara för att filtreras ner till en enda pjäs drag).

    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    own_piece_squares = {(r, c) for r, c in own_pieces if board[r][c] != ' '}

    controlled_squares = set()
    for (sr, sc), (er, ec) in moves:
        if (sr, sc) in own_piece_squares:
            controlled_squares.add((er, ec))

    return len(controlled_squares)

def three_fold_repetition(history, board, current_color, state):
    board_hash = (tuple(map(tuple, board)), current_color, _state_key(state))
    return history.get(board_hash, 0) >= 2

def Rook_Open_Files(board, color, state):
    """NY HEURISTIK: ger en liten bonus om man har torn på öppna linjer (dvs. inga egna bönder på den kolumnen)."""
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    pawn_piece = '♙' if color == 'white' else '♟'
    rook_piece = '♖' if color == 'white' else '♜'

    open_file_bonus = 0.0

    for r, c in own_pieces:
        if board[r][c] == rook_piece:
            # Kolla om det finns egna bönder på samma kolumn
            has_own_pawn_on_file = any(board[row][c] == pawn_piece for row in range(8))
            if not has_own_pawn_on_file:
                open_file_bonus += 0.2  # Bonus för varje torn på en öppen linje

    return open_file_bonus if color == 'white' else -open_file_bonus

def isolated_pawn_penalty_or_doubling(board, color, state):
    """NY HEURISTIK: ger en liten bonus om man har isolerade bönder (bönder som inte har några vänner på de intilliggande kolumnerna)."""
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    pawn_piece = '♙' if color == 'white' else '♟'

    isolated_pawn_penalty = 0.0

    for r, c in own_pieces:
        if board[r][c] == pawn_piece:
            # Kolla om det finns egna bönder på de intilliggande kolumnerna
            has_own_pawn_on_adjacent_files = any(
                board[row][col] == pawn_piece
                for row in range(8)
                for col in [c - 1, c + 1]
                if 0 <= col < 8
            )
            if not has_own_pawn_on_adjacent_files:
                isolated_pawn_penalty += 0.2  # Straff för varje isolerad bonde

    return isolated_pawn_penalty if color == 'white' else -isolated_pawn_penalty

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

def not_hanging_my_own_pieces(board, color, state, enemy_moves):
    """NY HEURISTIK: ger en liten bonus om man inte har hängande pjäser (pjäser som kan slås utan motdrag). Ju fler hängande pjäser, desto större straff."""
    own_pieces = state.white_pieces if color == 'white' else state.black_pieces
    enemy_color = 'black' if color == 'white' else 'white'
    
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
