import random
from clases import Prisoner, Box

def make_prisoners_and_box():
    amount = 101
    prisoners = [Prisoner(number=i, x_position=0, y_position=0, x_next_position=0, y_next_position=0, tries_left=50, next_box=0, current_box=0) for i in range(1, amount)]
    nodes = [Box(i) for i in range(1, 101)]
    shuffled_nodes = nodes[:]
    random.shuffle(shuffled_nodes)

    for i in range(100):
        nodes[i].next = shuffled_nodes[i].number
        
    return prisoners, nodes

def main():
    prisoners, room = make_prisoners_and_box()
    
    death = False
    succses = 0
    
    for prisoner in prisoners:
        box = room[prisoner.number - 1]
        prisoner.tries_left = 50
        
        while prisoner.tries_left != 0:
            prisoner.current_box = box.number
            prisoner.tries_left -= 1
            
            if box.next == prisoner.number:
                succses += 1
                break
                
            box = room[box.next - 1]
        else:
            death = True
            break
    
    if succses == 100:
        return True
    elif death:
        return False           
     
if __name__ == "__main__":
    amount = 1000
    succses = 0
    for i in range(amount):
        if main():
            succses += 1
    print(f"Success rate: {succses / amount * 100}%")