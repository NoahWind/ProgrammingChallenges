"""
The Tower of Hanoi

Move a tower of n disks from the source rod to the destination rod using an auxiliary rod.
Rules:
1. Only one disk can be moved at a time
2. A larger disk cannot be placed on top of a smaller disk
3. All disks must be stacked in ascending order (largest at bottom)
"""

class Tower:
    def __init__(self, name):
        self.name = name
        self.disks = []
        
    def add(self, disk):
        if self.disks and self.disks[-1] < disk:
            print(f"Error: Cannot place disk {disk} on top of {self.disks[-1]}")
            return False
        self.disks.append(disk)
        return True
        
    def remove(self):
        if not self.disks:
            return None
        return self.disks.pop()
        
    def __str__(self):
        return f"{self.name}: {self.disks}"

def solve_hanoi(n, source, auxiliary, destination):
    moves = []
    
    def hanoi(n, source, auxiliary, destination):
        if n == 1:
            disk = source.remove()
            destination.add(disk)
            moves.append((source.name, destination.name, disk))
            return
            
        hanoi(n-1, source, destination, auxiliary)
        hanoi(1, source, auxiliary, destination)
        hanoi(n-1, auxiliary, source, destination)
    
    hanoi(n, source, auxiliary, destination)
    return moves

def print_moves(moves):
    for i, (source, dest, disk) in enumerate(moves, 1):
        print(f"Move {i}: Move disk {disk} from {source} to {dest}")

if __name__ == "__main__":
    for n_disks in [3, 4]:
        source = Tower("A")
        auxiliary = Tower("B")
        destination = Tower("C")
        
        for disk in range(n_disks, 0, -1):
            source.add(disk)
            
        print(f"\nSolving Tower of Hanoi with {n_disks} disks:")
        print("Initial state:", source, auxiliary, destination)
        
        moves = solve_hanoi(n_disks, source, auxiliary, destination)
        print(f"\nSolution ({len(moves)} moves):")
        print_moves(moves)
        print("\nFinal state:", source, auxiliary, destination)
