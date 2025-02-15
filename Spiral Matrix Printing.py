""" A simulation that fills an n x n matrix in a spiral order.
Starting from the top-left corner, numbers are placed sequentially in a clockwise spiral pattern.
The matrix updates dynamically with a delay, clearing the terminal after each step for visualization.
The goal: Fill the matrix in a spiral order and display it dynamically.
"""

import math
import os
import time

i = int(input())

def print_matrix(matrix, offset):
    time.sleep(0.1)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    for row in matrix:
        print(" ".join(f"{num:>{offset}}" for num in row))
    
def matrix(i):
    numbers = []
    grid = []
    x_grid = []
    for j in range(1,i**2+1):
        numbers.append(j)
    
    offset = len(str(max(numbers)))

        
    for help in range(0,i):
        x_grid.append(0)

    for help in range(0,i):
        grid.append([0] * i)

    startx = 0
    starty = 0
    x = 0
    y = 0
    amount = math.ceil(i/2)
    for obj in numbers:
        grid[x][y]= obj
    
    lol = 0
    move_1 = i
    for i in range(amount):
        move_2 = move_1 - 1
        move_3 = move_2 - 1
        x = startx
        y = starty
        for j in range(0, move_1):
            obj = numbers[lol]
            if int(obj) < 10:
                obj = f"{obj} "
                
            grid[x][y]= obj
            y += 1
            lol += 1
            print_matrix(grid, offset)

        y -=1
        
        for j in range(move_2):
            x += 1
            obj = numbers[lol]
            
            grid[x][y]= obj
            lol += 1
            print_matrix(grid, offset)
        
        y -=1
        
        for j in range(move_2):
            obj = numbers[lol]
            
            grid[x][y]= obj
            y -= 1
            lol += 1
            print_matrix(grid, offset)  

        y +=1
        
        for j in range(move_3):
            obj = numbers[lol]
            
            x -= 1
            
            grid[x][y]= obj
            lol += 1
            print_matrix(grid, offset)
        
        startx += 1
        starty +=1
        move_1 -= 2
    
matrix(i)