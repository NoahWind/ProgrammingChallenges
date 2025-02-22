import random
from clases import Prisoner, Box

def make_prisoners_and_box():
    amount = 101
    prisoners = [Prisoner(number=i, x_position=0, y_position=0, x_next_position=0, y_next_position=0, tries_left=50, next_box=0, current_box=0) for i in range(1, amount)]
    room = [Box(number=i, next_number=0) for i in range(1, 101)]
    next = [i for i in range(1, 101)]
    random.shuffle(next)
    
    for box, next in zip(room, next):
        box.next_number = next
        
    return prisoners, room


def main():
    prisoners , room = make_prisoners_and_box()
    
    death = False
    
    for prisoner in prisoners:
        box = random.choice(room)
        prisoner.current_box = box.number
        prisoner.next_box = box.next_number
        while prisoner.tries_left != 0:
            prisoner.tries_left -= 1
            if box.number == prisoner.number:
                print("OMG")
                print(box.number)
                print(prisoner.number)
                break
            
            
            
            
                
        

if __name__ == "__main__":
    main()