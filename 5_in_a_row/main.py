width = 7
height = 6

width = 4
height =4
goal = 3


board = [[' ' for _ in range(width)] for _ in range(height)]

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
    valid_moves = moves(board)

    if not valid_moves:
        return False, None

    if player == main_player:
        best_move = None
        
        for move in valid_moves:
            x, y = move[0], move[1]
            board[y][x] = player

            if is_connect_4(board, player):
                win = True
            else:
                win, _ = solve_recursive(board, not alternating, main_player)

            board[y][x] = ' '

            if win:
                return True, move
            
            if best_move is None:
                best_move = move

        return False, best_move

    else:
        best_move = None
        for move in valid_moves:
            x, y = move[0], move[1]
            board[y][x] = player

            if is_connect_4(board, player):
                win = False
            else:
                win, _ = solve_recursive(board, not alternating, main_player)

            board[y][x] = ' '

            if not win:
                best_move = move
                break
            
            if best_move is None:
                best_move = move

        if best_move is not None:
            return False, best_move
        return True, None
                

main_player = 'x'


def play():
    while True:
        print_board(board)
        move = int(input("VÄLJ X KORDINAT FÖR DIG ATT LÄGGA PÅ "))

        options = moves(board)
        print(options)

        for obj in options:
            if obj[0] == move:
                move_x = obj[0]
                move_y = obj[1]


        board[move_y][move_x] = "o"
        alternating = True
        print_board(board)

        obj = solve_recursive(board, alternating, main_player)
        print(obj)
        move_x = obj[1][0]
        move_y = obj[1][1]

        board[move_y][move_x] = "x"

        if is_connect_4(board, "x"):
            print("X VANN")
            break
        elif is_connect_4(board, "o"):
            print("O VANN")
            break


        

play()