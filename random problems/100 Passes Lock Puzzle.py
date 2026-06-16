"""Simulerar 100 varv där cellers låsstatus växlas enligt ett mönster, och listar vilka celler som är öppna i slutet."""

prison_cells = []
for i in range (100):
    prison_cells.append(1)
temp = 1

while temp != len(prison_cells):
    cell = 0
    for i in prison_cells:
        
        if (cell + 1) % temp == 0:
            if(prison_cells[cell] == 1):
                prison_cells[cell] = 0
            else:
                prison_cells[cell] = 1
        cell +=1
    temp += 1
    
open_cells = []

for i, cell in enumerate(prison_cells, start=1):
    if cell == 0:
        open_cells.append(i)

print("Öppna celler i slutet:")
print(", ".join(str(num) for num in open_cells))
print(f"\Totalt antal öppna celler: {len(open_cells)}")
