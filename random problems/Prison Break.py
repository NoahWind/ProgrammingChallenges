"""
A prison break simulation where cells are locked (0) or unlocked (1).
Starting in the first cell, you free prisoners and flip all cell states (1 ↔ 0).
Continue until no more prisoners can be freed.
The goal: count the freed prisoners.
"""


prison = [int(x) for x in input().replace(',', ' ').split()]

def switsch(prison):
    list(prison)
    for i in range (0, len(prison)):
        if(prison[i]) == 1:
            prison[i] = 0
        else:
            prison[i] = 1
    return prison
        

def escape(prison):
    count = 0
    if prison[0] == 0:
        return 0
    
    x = 0
    for i in prison:
        x += 1
        if i == 1:
            count +=1
            switsch(prison)
    return count
        

print(escape(prison))