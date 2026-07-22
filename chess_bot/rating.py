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
OPEN_RATE = 0.6
END_RATE = 0.6

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

def evaluate_board(board, state, history, current_color):
    score = float(state.material_score)
    
    pos_key = get_position_hash(board, current_color, state)
    if (history.get(pos_key, 0) >= 2
            and abs(state.material_score) < 5
            and not _has_any_pawns(board)):
        return 0 
    
    a, b, c = get_game_phase_weights(state, OPEN_RATE, END_RATE)

    if state.half_move_clock > 80:
        if score < 0:
            score += (state.half_move_clock - 80) * 0.5 
        elif score > 0:
            score -= (state.half_move_clock - 80) * 0.5
    
    if is_king_safe(board, 'white', state): score += KING_SAFETY_BONUS * a + KING_SAFETY_BONUS * b + KING_SAFETY_BONUS * c
    if is_king_safe(board, 'black', state): score -= KING_SAFETY_BONUS * a + KING_SAFETY_BONUS * b + KING_SAFETY_BONUS * c

    if is_controling_center(board, 'white'): score += CONTROL_CENTER_BONUS  * a + CONTROL_CENTER_BONUS * b + CONTROL_CENTER_BONUS * c
    if is_controling_center(board, 'black'): score -= CONTROL_CENTER_BONUS * a + CONTROL_CENTER_BONUS * b + CONTROL_CENTER_BONUS * c

    if breakt_pawn_chains(board, 'white', state): score -= BREAKING_PAWN_CHAINS_BONUS * a + BREAKING_PAWN_CHAINS_BONUS * b + BREAKING_PAWN_CHAINS_BONUS * c
    if breakt_pawn_chains(board, 'black', state): score += BREAKING_PAWN_CHAINS_BONUS * a + BREAKING_PAWN_CHAINS_BONUS * b + BREAKING_PAWN_CHAINS_BONUS * c

    if bishop_pair(board, 'white', state): score += bishop_pair_bonus * a + bishop_pair_bonus * b + bishop_pair_bonus * c
    if bishop_pair(board, 'black', state): score -= bishop_pair_bonus * a + bishop_pair_bonus * b + bishop_pair_bonus * c

    if knight_on_the_rim(board, 'white', state): score -= knight_on_the_rim_penalty * a + knight_on_the_rim_penalty * b + knight_on_the_rim_penalty * c
    if knight_on_the_rim(board, 'black', state): score += knight_on_the_rim_penalty * a + knight_on_the_rim_penalty * b + knight_on_the_rim_penalty * c

    if how_many_squares_do_i_control(board, 'white', state): score += squares_controlled_bonus * how_many_squares_do_i_control(board, 'white', state) * a + squares_controlled_bonus * how_many_squares_do_i_control(board, 'white', state) * b + squares_controlled_bonus * how_many_squares_do_i_control(board, 'white', state) * c
    if how_many_squares_do_i_control(board, 'black', state): score -= squares_controlled_bonus * how_many_squares_do_i_control(board, 'black', state) * a + squares_controlled_bonus * how_many_squares_do_i_control(board, 'black', state) * b + squares_controlled_bonus * how_many_squares_do_i_control(board, 'black', state) * c

    pst_w, hang_w = evaluate_pieces_and_threats(board, state, 'white')
    pst_b, hang_b = evaluate_pieces_and_threats(board, state, 'black')

    score += pst_w * pieac_pos_bonus * a + pst_w * pieac_pos_bonus * b + pst_w * pieac_pos_bonus * c
    score += pst_b * pieac_pos_bonus * a + pst_b * pieac_pos_bonus * b + pst_b * pieac_pos_bonus * c
    score += hang_w * hanging_piece_penalty * a + hang_w * hanging_piece_penalty * b + hang_w * hanging_piece_penalty * c
    score += hang_b * hanging_piece_penalty * a + hang_b * hanging_piece_penalty * b + hang_b * hanging_piece_penalty * c


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