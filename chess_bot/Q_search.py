from rating import *
from rating import _terminal_score

DELTA_MARGIN = 2

def quiescence_search(board, state, alpha, beta, is_maximizing, history, q_depth=0, ENGINE_PARAMS=ENGINE_PARAMS):
    if time.time() - search_start_time > time_limit_seconds:
        raise TimeoutError()

    current_color = 'white' if is_maximizing else 'black'
    in_check = is_check(board, current_color, state)

    # --- Om man står i schack: inget "stand pat", måste svara på schacket ---
    if in_check:
        moves = get_all_legal_moves(current_color, board, state)
        if not moves:
            return _terminal_score(current_color, -q_depth)  # schackmatt

        # Ingen djupgräns-check här (q_depth >= 10) för schacksvar - forcerade
        # sekvenser (schack -> schack -> schack ...) är sällan djupa i praktiken
        # eftersom antalet lagliga svar oftast är litet, men vi sätter ändå
        # ett hårt tak för säkerhets skull.
        if q_depth >= 16:
            return evaluate_board(board, state, history, current_color, ENGINE_PARAMS)

        moves = order_moves(moves, board)

        if is_maximizing:
            best = float('-inf')
            for move in moves:
                record = apply_move(board, state, move[0], move[1])
                try:
                    val = quiescence_search(board, state, alpha, beta, False, history, q_depth + 1, ENGINE_PARAMS)
                finally:
                    undo_move(board, state, record)
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best
        else:
            best = float('inf')
            for move in moves:
                record = apply_move(board, state, move[0], move[1])
                try:
                    val = quiescence_search(board, state, alpha, beta, True, history, q_depth + 1, ENGINE_PARAMS)
                finally:
                    undo_move(board, state, record)
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best

    # --- Normalfallet: inte i schack -> stand pat + sök bara slag ---
    stand_pat = evaluate_board(board, state, history, current_color, ENGINE_PARAMS)

    if is_maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

    if q_depth >= 10:
        return stand_pat

    # captures_only=True: genererar ENDAST slag-pseudodrag från början,
    # istället för att räkna ut alla lagliga drag och sedan filtrera.
    capture_moves = get_all_legal_moves(current_color, board, state, captures_only=True)
    if not capture_moves:
        return stand_pat

    capture_moves = order_moves(capture_moves, board)

    if is_maximizing:
        for move in capture_moves:
            (sr, sc), (er, ec) = move
            moving_piece = board[sr][sc]
            captured = board[er][ec]
            captured_val = abs(PIECE_ORDER_VALUE.get(captured, 0))

            # Fånga bondeförvandling (värd en dam)
            if moving_piece in ('♙', '♟') and (er == 0 or er == 7):
                captured_val += 9
            # Fånga En Passant (värd en bonde)
            elif moving_piece in ('♙', '♟') and captured == ' ' and sc != ec:
                captured_val += 1
            if stand_pat + captured_val + DELTA_MARGIN <= alpha:
                continue

            # SEE: hoppa över klart förlustaffärer (t.ex. Dxb2 mot en
            # välförsvarad bonde) - de är i praktiken aldrig rätt i tyst läge.
            if static_exchange_eval(board, move) < 0:
                continue

            record = apply_move(board, state, move[0], move[1])
            try:
                eval_val = quiescence_search(board, state, alpha, beta, False, history, q_depth + 1, ENGINE_PARAMS)
            finally:
                undo_move(board, state, record)
            if eval_val >= beta:
                return beta
            alpha = max(alpha, eval_val)
        return alpha
    else:
        for move in capture_moves:
            (sr, sc), (er, ec) = move
            moving_piece = board[sr][sc]
            captured = board[er][ec]
            captured_val = abs(PIECE_ORDER_VALUE.get(captured, 0))

            # Fånga bondeförvandling (värd en dam)
            if moving_piece in ('♙', '♟') and (er == 0 or er == 7):
                captured_val += 9
            # Fånga En Passant (värd en bonde)
            elif moving_piece in ('♙', '♟') and captured == ' ' and sc != ec:
                captured_val += 1
            if stand_pat - captured_val - DELTA_MARGIN >= beta:
                continue

            if static_exchange_eval(board, move) < 0:
                continue

            record = apply_move(board, state, move[0], move[1])
            try:
                eval_val = quiescence_search(board, state, alpha, beta, True, history, q_depth + 1, ENGINE_PARAMS)
            finally:
                undo_move(board, state, record)
            if eval_val <= alpha:
                return alpha
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

def minimax(board, state, depth, alpha, beta, is_maximizing, history, local_cache, ENGINE_PARAMS=ENGINE_PARAMS):
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

    # Använder lokala cachen
    if use_cache and board_hash in local_cache:
        cached_eval, cached_depth, node_type = local_cache[board_hash]
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
        result = quiescence_search(board, state, alpha, beta, is_maximizing, history, 0, ENGINE_PARAMS)
        history[board_hash] -= 1
        return result

    moves = order_moves(moves, board)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            record = apply_move(board, state, move[0], move[1])
            try:
                # Skickar med local_cache rekursivt
                eval_val = minimax(board, state, depth - 1, alpha, beta, not is_maximizing, history, local_cache, ENGINE_PARAMS)
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
            local_cache[board_hash] = (max_eval, depth, node_type)
        history[board_hash] -= 1
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            record = apply_move(board, state, move[0], move[1])
            try:
                # Skickar med local_cache rekursivt
                eval_val = minimax(board, state, depth - 1, alpha, beta, not is_maximizing, history, local_cache, ENGINE_PARAMS)
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
            local_cache[board_hash] = (min_eval, depth, node_type)
        history[board_hash] -= 1
        return min_eval
    
def get_best_move(board, depth, color, state, history=None, params=None, DEFAULT_TIME_LIMIT_SECONDS=5):
    global search_start_time, time_limit_seconds

    search_start_time = time.time()
    time_limit_seconds = DEFAULT_TIME_LIMIT_SECONDS  # Aktivera alltid ordinarie tänktid

    is_white_turn = (color == 'white')

    if history is None:
        history = {}
        
    if params is None:
        params = ENGINE_PARAMS
        
    local_cache = {}  # NY LOKAL CACHE FÖR DENNA SÖKNING
        
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
            
            alpha = float('-inf')
            beta = float('inf')

            for move in moves:
                record = apply_move(board, state, move[0], move[1])
                try:
                    # Skickar med local_cache hit ner
                    move_value = minimax(board, state, current_depth - 1, alpha, beta, not is_white_turn, history, local_cache, ENGINE_PARAMS=params)
                finally:
                    undo_move(board, state, record)

                if color == 'white':
                    if move_value > best_value_for_this_depth:
                        best_value_for_this_depth = move_value
                        best_move_for_this_depth = move
                    alpha = max(alpha, move_value)
                else:
                    if move_value < best_value_for_this_depth:
                        best_value_for_this_depth = move_value
                        best_move_for_this_depth = move
                    beta = min(beta, move_value)

            if best_move_for_this_depth is not None:
                best_move_overall = best_move_for_this_depth
                best_value = best_value_for_this_depth

    except TimeoutError:
        print("Tiden tog slut – använder resultat från föregående färdiga djup.")
    
    # Använd local_cache för utskrifterna
    print("Cache size:", len(local_cache), "entries")
    print("in MB:", len(local_cache) * 64 / (1024 * 1024))
    print("Best move found:", best_move_overall, "with evaluation:", best_value)
    
    return best_move_overall