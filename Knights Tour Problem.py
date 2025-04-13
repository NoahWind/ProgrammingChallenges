"""
The Knight's Tour Problem

Given an N x N chessboard, place a knight on any starting position and find a sequence of moves
that allows the knight to visit every square exactly once. The knight moves according to standard chess rules:
two squares in one direction and then one square perpendicular to that direction.
"""

def is_valid_move(x, y, board_size, board):
    return (0 <= x < board_size and 
            0 <= y < board_size and 
            board[x][y] == 0)

def solve_knights_tour(board_size, start_x=0, start_y=0):
    moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
    
    board = [[0] * board_size for _ in range(board_size)]
    move_count = 1
    board[start_x][start_y] = move_count
    
    def find_tour(curr_x, curr_y, move_count):
        if move_count == board_size * board_size:
            return True
            
        for dx, dy in moves:
            next_x, next_y = curr_x + dx, curr_y + dy
            
            if is_valid_move(next_x, next_y, board_size, board):
                board[next_x][next_y] = move_count + 1
                if find_tour(next_x, next_y, move_count + 1):
                    return True
                board[next_x][next_y] = 0
                
        return False
    
    if find_tour(start_x, start_y, move_count):
        return board
    return None

def print_board(board):
    if not board:
        print("No solution exists!")
        return
        
    for row in board:
        print(" ".join(f"{num:2d}" for num in row))

if __name__ == "__main__":
    board_sizes = [5, 6]
    for size in board_sizes:
        print(f"\nKnight's Tour on {size}x{size} board:")
        solution = solve_knights_tour(size)
        print_board(solution)
