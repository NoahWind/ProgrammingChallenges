"""
The Water Jug Problem

You have two jugs with capacities X and Y liters (whole numbers).
Neither has any measuring marks on it. You can:
1. Fill a jug completely
2. Empty a jug completely
3. Pour from one jug to the other until either the first is empty or the second is full

The goal is to measure exactly Z liters of water using these jugs.
"""

from collections import deque

class State:
    def __init__(self, x, y, steps=None):
        self.x = x
        self.y = y
        self.steps = steps or []
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

def solve_water_jug(capacity_x, capacity_y, target):
    visited = set()
    queue = deque([State(0, 0)])
    
    while queue:
        current = queue.popleft()
        
        if current.x == target or current.y == target:
            return current.steps
            
        if (current.x, current.y) in visited:
            continue
            
        visited.add((current.x, current.y))
        
        if current.x < capacity_x:
            new_steps = current.steps + [f"Fill jug X: ({capacity_x}, {current.y})"]
            queue.append(State(capacity_x, current.y, new_steps))
            
        if current.y < capacity_y:
            new_steps = current.steps + [f"Fill jug Y: ({current.x}, {capacity_y})"]
            queue.append(State(current.x, capacity_y, new_steps))
            
        if current.x > 0:
            new_steps = current.steps + [f"Empty jug X: (0, {current.y})"]
            queue.append(State(0, current.y, new_steps))
            
        if current.y > 0:
            new_steps = current.steps + [f"Empty jug Y: ({current.x}, 0)"]
            queue.append(State(current.x, 0, new_steps))
            
        if current.x > 0 and current.y < capacity_y:
            amount = min(current.x, capacity_y - current.y)
            new_steps = current.steps + [f"Pour X to Y: ({current.x - amount}, {current.y + amount})"]
            queue.append(State(current.x - amount, current.y + amount, new_steps))
            
        if current.y > 0 and current.x < capacity_x:
            amount = min(current.y, capacity_x - current.x)
            new_steps = current.steps + [f"Pour Y to X: ({current.x + amount}, {current.y - amount})"]
            queue.append(State(current.x + amount, current.y - amount, new_steps))
    
    return None

def print_solution(steps):
    if not steps:
        print("No solution exists!")
        return
        
    print(f"Solution found in {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

if __name__ == "__main__":
    solution = solve_water_jug(4, 3, 2)
    print_solution(solution)
    
    print("\nTrying another example:")
    solution = solve_water_jug(5, 3, 4)
    print_solution(solution)
