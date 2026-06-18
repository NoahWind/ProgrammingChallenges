import random

sudoku_board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

sudoku_board_copy = sudoku_board

def get_free_cors(board):
    x = 0
    y = 0
    free_cords = []
    for j in board:
        for i in j:
            if board[y][x] == 0:
                free_cords.append([x,y])
            x+=1
        y+=1
        x=0

    return free_cords

box = 3
width = box * 3

def check_row(y_cord, number):
    row = sudoku_board_copy[y_cord]

    if number in row:
        return False
    else:
        return True

def check_line(x_cord, number):
    for i in range(width):
        if number == sudoku_board_copy[i][x_cord]:
            return False
    return True

def check_box(x_cord, y_cord, number):
    start_x = (x_cord // 3) * 3
    start_y = (y_cord // 3) * 3

    for j in range(start_y, start_y + 3):
        for i in range(start_x, start_x + 3):
            if sudoku_board_copy[j][i] == number:
                return False
    return True

def solve(board, free_cord):
    if not free_cord:
        print(board)
        return True
    cord = free_cord.pop(0)
    x = cord[0]
    y = cord[1]
    for i in range(1,10):
        if check_box(x,y,i) and check_line(x,i) and check_row(y,i):
            board[y][x] = i
            if solve(board,free_cord):
                return True
            board[y][x] = 0
    free_cord.insert(0, cord)
    return False
    


solve(sudoku_board_copy, get_free_cors(sudoku_board_copy))