import random
import math

node_amount = 200
neighbours_max = 4
X = 100
Y = 100

nodes = []
edges = []

ID, NX, NY, NEIGHBORS = 0, 1, 2, 3
FROM_NODE, TO_NODE, WEIGHT = 0, 1, 2

for obj in range(node_amount):
    node = [
        obj,                   
        random.randint(0, X), 
        random.randint(0, Y),   
        []       
    ]
    
    if len(nodes) > 0:
        kandidater = list(nodes)
        random.shuffle(kandidater)
        
        onskat_antal = random.randint(1, 2)
        aktuella_kopplingar = 0
        
        for potentiell_granne in kandidater:
            if aktuella_kopplingar >= onskat_antal or len(node[NEIGHBORS]) >= neighbours_max:
                break
                
            if len(potentiell_granne[NEIGHBORS]) < neighbours_max:
                node[NEIGHBORS].append(potentiell_granne[ID])
                potentiell_granne[NEIGHBORS].append(obj)
                dx = node[NX] - potentiell_granne[NX]
                dy = node[NY] - potentiell_granne[NY]
                avstand = round(math.sqrt(dx**2 + dy**2), 2)
                
                edges.append([obj, potentiell_granne[ID], avstand])
                
                aktuella_kopplingar += 1

    nodes.append(node)

print("--- EDGES (KANTER MED VIKT) ---")
print(f"{'Från':<6} {'Till':<6} {'Vikt (Avstånd)':<15}")
print("-" * 30)
for e in edges:
    print(f"{e[FROM_NODE]:<6} {e[TO_NODE]:<6} {e[WEIGHT]:<15}")

def alredy_been(options, passed):
    options_left = []

    for obj in options:
        was_visited = False  
        
        for thing in passed:
            if obj[1] == thing[0]:
                was_visited = True 
                break       
                
        if not was_visited:  
            options_left.append(obj)
    return(options_left)

def sort_after_shortest(current):
    options = []
    for obj in edges:
        #print(obj[0], obj[1], current)

        if obj[0] == current:
            options.append(obj)
        if obj[1] == current:
            options.append([obj[1], obj[0], obj[2]])
    options = sorted(options, key=lambda x: x[2])
    return options

def A_start(current, passed, start,end):
    if current==end:
        print("We have reached the end!") 
        print("Path taken:")
        for obj in passed:
            print(f"Node ID: {obj[0]}, Coordinates: ({obj[1]}, {obj[2]})")
        return True

    optoins = alredy_been(sort_after_shortest(current), passed)
    for obj in optoins:
        nasta_nod = obj[1]
        
        nod_objekt = nodes[nasta_nod] 
        passed.append(nod_objekt)
        if A_start(obj[1], passed, start,end):
            return True
        passed.pop(-1)
    return False
    

def main():

    passed = []

    start = random.randint(0, node_amount - 1)
    end = random.randint(0, node_amount - 1)
    passed.append(nodes[start])

    while end == start:
        end = random.randint(0, node_amount - 1)
    print(f"Start node: {start}, End node: {end}")
    A_start(start, passed, start, end)
    

main()