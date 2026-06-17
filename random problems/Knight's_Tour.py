import random
import math
import time

board = []
letters = ["a","b","c","d","e","f","g","h"]
passed_in_print = []
knight = "♘"
unthuched_squres = 8*8

def make_bord():
    for obj in letters:
        i=1

        for x in range(len(letters)):
            board.append([obj,i])
            i+=1

def print_real_board(knight_pos):
    global passed_in_print
    rows = [8, 7, 6, 5, 4, 3, 2, 1]
    cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
    
    print(f"\n--- Drag: {knight_pos} ---")
    for y in rows:
        row_string = ""
        for x in cols:
            if [x, y] == knight_pos:
                row_string += "♘ "
                passed_in_print.append(knight_pos)
            elif [x,y] in passed_in_print:
                row_string += "x "
            else:
                row_string += ". "
        print(f"{y}  {row_string}")
        
    print("   a b c d e f g h")

def sort_after_Warnsdorffs(positions):
    letters = ["a","b","c","d","e","f","g","h"]
    middle = 4.5
    
    def get_distance(pos):
        x_index = letters.index(pos[0]) + 1
        y_val = pos[1]
        dis = math.sqrt((x_index - middle)**2 + (y_val - middle)**2)
        return dis
    positions.sort(key=get_distance, reverse=True)
    
    return positions
    

def get_leagel_moves(position):
    letters = ["a","b","c","d","e","f","g","h"]
    dir = 4
    x_posible_pos = []
    current_x_pos = position[0]
    current_x_pos_in_list = letters.index(current_x_pos)

    directions = [-2,-1,1,2]

    for dir in directions:
        if current_x_pos_in_list + int(dir)>=0 and current_x_pos_in_list + int(dir)<8:
            if dir == -2 or dir ==2:
                y_change = 1
                for g in [-1,1]:
                    if position[1] + int(y_change*g)>0 and position[1] + int(y_change*g)<9:
                        x_posible_pos.append([letters[current_x_pos_in_list + int(dir)], position[1] + int(y_change*g)])

            else:
                y_change = 2
                for g in [-1,1]:
                    if position[1] + int(y_change*g)>0 and position[1] + int(y_change*g)<9:
                        x_posible_pos.append([letters[current_x_pos_in_list + int(dir)], position[1] + int(y_change*g)])

    return(x_posible_pos)  

finished = []

def move_knight(position, passed:list, unthuched_squres):
    global finished 
    
    passed.append(position)
    possible_positions = sort_after_Warnsdorffs(get_leagel_moves(position))
    unthuched_squres -= 1
    
    if unthuched_squres == 0:
        finished = passed.copy() 
        return True
        
    for obj in possible_positions:
        if obj in passed:
            pass
        else:
            if move_knight(obj, passed, unthuched_squres):
                return True
                
    passed.pop()
    return False


make_bord()
move_knight(random.choice(board), [], unthuched_squres)

print(board)
for move in finished:
    print_real_board(move)
    print("_________________________________")
    time.sleep(0.3)