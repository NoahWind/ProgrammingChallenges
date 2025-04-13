"""
Langton's Ant

Simulate Langton's ant, a cellular automaton on a grid with simple rules:
1. At a white square, turn right 90°, flip the color to black, move forward
2. At a black square, turn left 90°, flip the color to white, move forward

The ant creates complex emergent behavior from these simple rules.
"""

import os
import time

class LangtonAnt:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0] * width for _ in range(height)]
        self.ant_x = width // 2
        self.ant_y = height // 2
        self.direction = 0
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
    def step(self):
        current_color = self.grid[self.ant_y][self.ant_x]
        
        if current_color == 0:
            self.direction = (self.direction + 1) % 4
            self.grid[self.ant_y][self.ant_x] = 1
        else:
            self.direction = (self.direction - 1) % 4
            self.grid[self.ant_y][self.ant_x] = 0
            
        dx, dy = self.directions[self.direction]
        self.ant_x = (self.ant_x + dx) % self.width
        self.ant_y = (self.ant_y + dy) % self.height
        
    def display(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        for y in range(self.height):
            for x in range(self.width):
                if x == self.ant_x and y == self.ant_y:
                    print('*', end=' ')
                else:
                    print('█' if self.grid[y][x] else '·', end=' ')
            print()

def run_simulation(width, height, steps, delay=0.1):
    ant = LangtonAnt(width, height)
    
    for _ in range(steps):
        ant.display()
        ant.step()
        time.sleep(delay)

if __name__ == "__main__":
    run_simulation(30, 20, 200, 0.1)
