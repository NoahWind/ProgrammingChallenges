import random
import math

node_amount = 50
neighbours_max = 4
X = 100
Y = 100

nodes = []
edges = []

import matplotlib.pyplot as plt

# Här sparar vi en "snapshot" av varje steg algoritmen tar
search_history = []

def spela_upp_sokning(start, end, hittade_vag):
    plt.ion() # Starta interaktivt läge för animation
    plt.figure(figsize=(9, 9))
    
    # Lista för att hålla koll på alla unika noder vi hittills har testat (för att färga dem gula)
    besökta_noder = set()

    for steg_nummer, steg in enumerate(search_history):
        plt.clf() # Rensa skärmen för nästa bildruta
        
        aktiv_nod = steg['current']
        aktuell_vag = steg['passed']
        besökta_noder.add(aktiv_nod)
        
        # 1. Rita alla nätverkets kanter i ljusgrått
        for edge in edges:
            n1 = nodes[edge[0]]
            n2 = nodes[edge[1]]
            plt.plot([n1[NX], n2[NX]], [n1[NY], n2[NY]], color="#e0e0e0", zorder=1)
            
        # 2. Rita alla noder som hittills har prövats (Gula)
        for node_id in besökta_noder:
            n = nodes[node_id]
            plt.scatter(n[NX], n[NY], color="gold", s=150, zorder=2)
            
        # 3. Rita den aktiva vägen just i detta steg (Grön linje)
        if len(aktuell_vag) > 1:
            for i in range(len(aktuell_vag) - 1):
                n1 = aktuell_vag[i]
                n2 = aktuell_vag[i+1]
                plt.plot([n1[NX], n2[NX]], [n1[NY], n2[NY]], color="limegreen", linewidth=3, zorder=3)
                
        # 4. Rita alla vanliga noder (Blå) och deras ID-nummer
        for n in nodes:
            plt.scatter(n[NX], n[NY], color="royalblue", s=80, zorder=4)
            plt.text(n[NX]+1, n[NY]+1, str(n[ID]), fontsize=9, weight='bold')

        # 5. Markera Start (Grön) och Mål (Röd) extra stort
        ns = nodes[start]
        ne = nodes[end]
        plt.scatter(ns[NX], ns[NY], color="limegreen", s=200, edgecolors="black", zorder=5)
        plt.scatter(ne[NX], ne[NY], color="crimson", s=200, edgecolors="black", zorder=5)
        
        # 6. Markera var algoritmens "huvud" stod i detta steg (Orange ring)
        if aktiv_nod is not None:
            nc = nodes[aktiv_nod]
            plt.scatter(nc[NX], nc[NY], color="orange", s=250, edgecolors="black", linewidth=2, zorder=6)

        plt.title(f"Repris av sökningen (Steg {steg_nummer + 1}/{len(search_history)})\nGul = Prövad | Grön linje = Aktiv sökstig")
        plt.xlim(-5, X + 5)
        plt.ylim(-5, Y + 5)
        
        plt.pause(0.3) # Hastighet på animationen (0.3 sekunder per steg)

    # När hela historiken visats, visa slutresultatet permanent
    plt.clf()
    # Rita om allt en sista gång med den slutgiltiga stigen om vi hittade en
    for edge in edges:
        plt.plot([nodes[edge[0]][NX], nodes[edge[1]][NX]], [nodes[edge[0]][NY], nodes[edge[1]][NY]], color="#ddd", zorder=1)
    for n in nodes:
        plt.scatter(n[NX], n[NY], color="royalblue", s=80, zorder=3)
        plt.text(n[NX]+1, n[NY]+1, str(n[ID]), fontsize=9)
    
    if hittade_vag:
        plt.title(f"🎉 KLAR! Repris färdig. Väg hittades från {start} till {end}!")
        for i in range(len(aktuell_vag) - 1):
            plt.plot([aktuell_vag[i][NX], aktuell_vag[i+1][NX]], [aktuell_vag[i][NY], aktuell_vag[i+1][NY]], color="limegreen", linewidth=4, zorder=2)
    else:
        plt.title(f"❌ Repris färdig. Ingen väg kunde hittas mellan {start} och {end}.")
        
    plt.scatter(nodes[start][NX], nodes[start][NY], color="limegreen", s=150, edgecolors="black", zorder=4)
    plt.scatter(nodes[end][NX], nodes[end][NY], color="crimson", s=150, edgecolors="black", zorder=4)
    
    plt.ioff()
    plt.show()

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

def A_start(current, passed, start, end):
    # Spara en ögonblicksbild av det här steget i historiken (kopiera listan med list())
    search_history.append({'current': current, 'passed': list(passed)})

    if current == end:
        print("DONE")
        return True

    options = alredy_been(sort_after_shortest(current), passed)
    for obj in options:
        nasta_nod = obj[1] 
        
        nod_objekt = nodes[nasta_nod] 
        passed.append(nod_objekt)
        
        if A_start(obj[1], passed, start, end):
            return True
            
        passed.pop(-1)
        # Spara även steget när vi backtrackar (så vi ser att den backar!)
        search_history.append({'current': current, 'passed': list(passed)})
        
    return False

def main():

    passed = []
    search_history = []

    start = random.randint(0, node_amount - 1)
    end = random.randint(0, node_amount - 1)
    passed.append(nodes[start])

    while end == start:
        end = random.randint(0, node_amount - 1)
    print(f"Start node: {start}, End node: {end}")
    hittade_vag = 0
    hittade_vag = A_start(start, passed, start, end)
    spela_upp_sokning(start, end, hittade_vag)
    

main()