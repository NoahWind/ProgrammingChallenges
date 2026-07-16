import os
import sqlite3

width = 7
height = 6
size = width * height

goal = 4

board = [[' ' for _ in range(width)] for _ in range(height)]

calculated_positions = {}

db_conn = None

def get_conn():
    global db_conn
    if db_conn is None:
        db_conn = sqlite3.connect("connect4_perfect_db.sqlite3")
    return db_conn

def board_to_key(board, alternating):
    code = 0
    for c in range(width):
        for r in range(height):
            token = board[r][c]
            if token == 'o':
                code |= 1 << ((c * height + r) * 2)
            elif token == 'x':
                code |= 2 << ((c * height + r) * 2)
    dataset_alternating = not alternating
    return (code << 1) | (1 if dataset_alternating else 0)

def add_to_calc_from_json():
    if os.path.exists("connect4_perfect_db.sqlite3"):
        print("Databas hittad, slås upp vid behov under spelets gång.")
    else:
        print("Ingen databas hittad, boten kör utan förberäknade positioner.")


def is_position_alredy_calculated(position, alternating):
    board_tuple = tuple(tuple(row) for row in board)
    key = (board_tuple, alternating)
    if key in calculated_positions:
        return calculated_positions[key]

    if not os.path.exists("connect4_perfect_db.sqlite3"):
        return None

    conn = get_conn()
    db_key = board_to_key(board, alternating)
    row_data = conn.execute(
        "SELECT score, col, row FROM positions WHERE key = ?", (str(db_key),)
    ).fetchone()

    if row_data is None:
        return None

    score, col, row = row_data
    move = (col, row) if col is not None else None
    result = (score, move)
    calculated_positions[key] = result
    return result

def add_to_calc(board, alternating, result):
    board_tuple = tuple(tuple(row) for row in board)
    key = (board_tuple, alternating)
    calculated_positions[key] = result

def free_spaces_left(board):
    amount = 0
    for x in range(width):
        for y in range(height):
            if board[y][x] == ' ':
                amount += 1
    return amount

def is_connect_4(board, turn):
    for r in range(height):
        for c in range(width - goal + 1):
            if all(board[r][c+i] == turn for i in range(goal)):
                return True

    for c in range(width):
        for r in range(height - goal + 1):
            if all(board[r+i][c] == turn for i in range(goal)):
                return True

    for r in range(height - goal + 1):
        for c in range(width - goal + 1):
            if all(board[r+i][c+i] == turn for i in range(goal)):
                return True

    for r in range(goal - 1, height):
        for c in range(width - goal + 1):
            if all(board[r-i][c+i] == turn for i in range(goal)):
                return True

    return False

def sort_closest_to_center(moves):
    center = width // 2
    return sorted(moves, key=lambda move: abs(move[0] - center))


def print_board(board):
    for row in board[::-1]:
        print(row)

def moves(board):
    valid_moves = []
    width = len(board[0])
    height = len(board)

    for x in range(width):
        for y in range(height):
            if board[y][x] == ' ':
                valid_moves.append((x, y))
                break

    return valid_moves

def solve_recursive(board, alternating, main_player):

    player = 'x' if alternating else 'o'
    opponent = 'o' if alternating else 'x'

    if free_spaces_left(board) > size - 5:
        place = width // 2
        for y in range(height):
            if board[y][place] == ' ':
                return 0, (place, y)

    valid_moves = moves(board)
    valid_moves = sort_closest_to_center(valid_moves)

    cached = is_position_alredy_calculated(board, alternating)

    if cached is not None:
        return cached

    for move in valid_moves:
        board[move[1]][move[0]] = player
        if is_connect_4(board, player):
            board[move[1]][move[0]] = ' '
            add_to_calc(board, alternating, (1, move))
            return 1, move
        board[move[1]][move[0]] = ' '

    if not valid_moves:
        return 0, None

    for move in valid_moves:
        board[move[1]][move[0]] = opponent
        opponent_wins_here = is_connect_4(board, opponent)
        board[move[1]][move[0]] = ' '

        if opponent_wins_here:
            board[move[1]][move[0]] = player
            score, _ = solve_recursive(board, not alternating, main_player)
            board[move[1]][move[0]] = ' '

            result = (-score, move)
            add_to_calc(board, alternating, result)
            return result

    non_losing_moves = []
    for move in valid_moves:
        board[move[1]][move[0]] = player
        score, _ = solve_recursive(board, not alternating, main_player)
        board[move[1]][move[0]] = ' '
        if -score == 1:
            add_to_calc(board, alternating, (1, move))
            return 1, move
        if score != 1:
            non_losing_moves.append(move)

    if non_losing_moves:
        result = (0, non_losing_moves[0])
        add_to_calc(board, alternating, result)
        return result

    safe_moves = []
    for move in valid_moves:
        board[move[1]][move[0]] = player
        opponent_can_win = False
        for reply in moves(board):
            board[reply[1]][reply[0]] = opponent
            if is_connect_4(board, opponent):
                opponent_can_win = True
            board[reply[1]][reply[0]] = ' '
            if opponent_can_win:
                break
        board[move[1]][move[0]] = ' '
        if not opponent_can_win:
            safe_moves.append(move)

    result = (-1, safe_moves[0] if safe_moves else valid_moves[0])
    add_to_calc(board, alternating, result)
    return result

def play():
    import time

    add_to_calc_from_json()

    print("Boten spelar mot sig själv! 'o' vs 'x'.")
    
    # Vi behöver hålla koll på vems tur det är
    turn = 'o' 

    while True:
        amout_new_positions = len(calculated_positions)
        print(f"Antal beräknade positioner: {amout_new_positions}")
        print_board(board)
        
        # 1. Boten tänker
        start = time.time()
        print(f"Boten ({turn}) tänker...")
        
        # Vi skickar in vems tur det är till solve_recursive
        # Om turn är 'o', skicka in False (eftersom din kod antar 'x' vid True)
        is_x_turn = (turn == 'x')
        _, best_move = solve_recursive(board, is_x_turn, turn)
        
        # 2. Utför draget
        if best_move:
            board[best_move[1]][best_move[0]] = turn
        else:
            print("Inga drag kvar, oavgjort eller slut!")
            break
            
        end = time.time()
        print(f"Boten ({turn}) gjorde sitt drag på {end - start:.2f} sekunder.")

        # 3. Kontrollera vinst
        if is_connect_4(board, turn):
            print_board(board)
            print(f"Boten {turn} vann!")
            break
            
        # 4. Byt tur
        turn = 'x' if turn == 'o' else 'o'
        newly_calculated_positions = len(calculated_positions) - amout_new_positions
        print(f"Antal nya positioner beräknade: {newly_calculated_positions}")

        # Valfritt: lägg till en liten paus för att hinna se vad som händer
        time.sleep(0.5)

play()

"""        print_board(board)
        move = int(input("Välj X-koordinat för 'o': "))
        print(f"Antal beräknade positioner: {len(calculated_positions)}")
        options = moves(board)

        found = False
        for x, y in options:
            if x == move:
                board[y][x] = "o"
                found = True
                break
        if not found:
            print("Ogiltigt drag, försök igen!")
            continue"""