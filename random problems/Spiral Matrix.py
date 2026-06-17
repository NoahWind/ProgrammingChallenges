import math

def get_center_of_matrix(size):
    if size % 2 == 0:
        return size // 2 - 1, size // 2 - 1
    return size // 2, size // 2

def fill_spiral(grid, size, x, y, list_of_letters):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    grid[x][y] = list_of_letters.pop(0)
    
    step = 1
    while list_of_letters:
        for _ in range(2):
            dx, dy = directions.pop(0)
            directions.append((dx, dy))
            for _ in range(step):
                x += dx
                y += dy
                if 0 <= x < size and 0 <= y < size and list_of_letters:
                    grid[x][y] = list_of_letters.pop(0)
        step += 1

    return grid

def spiralMatrix(size, word):
    list_of_letters = list(word[:size**2])
    
    while len(list_of_letters) < size**2:
        list_of_letters.append("+")
    
    grid = [["+"] * size for _ in range(size)]

    x, y = get_center_of_matrix(size)

    grid = fill_spiral(grid, size, x, y, list_of_letters)

    for row in grid:
        print(row)

spiralMatrix(3, "COPYRIGHTS")
spiralMatrix(5, "SUPERLUMBERJACK")
