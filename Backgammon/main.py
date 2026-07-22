import random

HOME_WHITE = 0
HOME_BLACK = 0

OUT_WHITE = 0
OUT_BLACK = 0

board = [-2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2]


BLOCK = 2

DEPTH = 100
AMMOUNT = 100

def Get_Positions(board, player):
    possitions = []

    x = 0
    for obj in board:
        if obj != 0:
            if player == -1 and obj < 0:
                possitions.append([obj,x])
            if player == 1 and obj > 0:
                possitions.append([obj,x])

        x += 1
    return possitions

def get_amount_of_1s(board, player):
    pass

def get_enemy_double_pos(board, player):
    enemy_positions = Get_Positions(board, -player)
    double_positions = []
    if player == 1:
        for obj in enemy_positions:
            if obj[0] <= -2:
                double_positions.append(obj[1])
    if player == -1:
        for obj in enemy_positions:
            if obj[0] >= 2:
                double_positions.append(obj[1])
    return double_positions

def Get_Leagle_Moves(dice_1, dice_2, board, player):
    moves = []
    possitions = Get_Positions(board,player)
    enemy_double_pos = get_enemy_double_pos(board, player)
    Range_min = 0
    Range_max = len(board)-1

    for obj in possitions:
        if obj[1] + player * dice_1 < Range_max and obj[1] + player * dice_1 > Range_min:
            if obj[1] + player * dice_1 not in enemy_double_pos:
                moves.append([obj[1], obj[1] + player * dice_1, dice_1])
        if obj[1] + player * dice_2 < Range_max and obj[1] + player * dice_2 > Range_min:
            if obj[1] + player * dice_2 not in enemy_double_pos:
                moves.append([obj[1], obj[1] + player * dice_2, dice_2])

    
    return moves

def rate_move(move):
    score = 0
    #Lämna dem ensamma
    #Nära mål
    # Kan man ta nåns pjäs
    return score

print(Get_Leagle_Moves(random.randint(1, 6), random.randint(1, 6), board, 1))
print(get_enemy_double_pos(board, 1))