"""
The N Queens Problem

Place N chess queens on an N×N chessboard so that no two queens threaten each other.
A solution requires that no two queens share the same row, column, or diagonal.
"""

def is_safe(board, row, col, n):
    for i in range(col):
        if board[row][i] == 1:
            return False
            
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
            
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
            
    return True

def solve_n_queens(n):
    board = [[0 for x in range(n)] for y in range(n)]
    
    def solve_util(col):
        if col >= n:
            return True
            
        for i in range(n):
            if is_safe(board, i, col, n):
                board[i][col] = 1
                
                if solve_util(col + 1):
                    return True
                    
                board[i][col] = 0
                
        return False
    
    if solve_util(0) == False:
        return None
    return board

def print_solution(board):
    if not board:
        print("No solution exists!")
        return
        
    for row in board:
        print(" ".join("Q" if x == 1 else "." for x in row))

if __name__ == "__main__":
    board_sizes = [4, 8, 12]
    for n in board_sizes:
        print(f"\nSolution for {n} queens:")
        solution = solve_n_queens(n)
        print_solution(solution)
