import random

node_amount = 10
neighbours_max = 4
X = 100
Y = 100

nodes = []
edge = []

for obj in range(node_amount):
    node = {
        "id": obj,
        "x": random.randint(0, X),
        "y": random.randint(0, Y),
        "neighbors": []
    }
    
    if len(nodes) > 0:
        kandidater = list(nodes)
        random.shuffle(kandidater)
        
        onskat_antal = random.randint(1, 2)
        aktuella_kopplingar = 0
        
        for potentiell_granne in kandidater:
            if aktuella_kopplingar >= onskat_antal or len(node["neighbors"]) >= neighbours_max:
                break
            if len(potentiell_granne["neighbors"]) < neighbours_max:
                node["neighbors"].append(potentiell_granne["id"])
                potentiell_granne["neighbors"].append(obj)
                
                aktuella_kopplingar += 1

    nodes.append(node)

for n in nodes:
    print(n)