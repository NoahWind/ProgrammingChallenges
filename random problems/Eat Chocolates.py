"""
    Calculates chocolates Arun can eat, including free ones from every 3 wrappers.
    Extracts numbers from input, validates them, and simulates the process.
    Returns total chocolates or "Invalid Input" for invalid cases.
"""

def count_eatable_chocolates(a, b):
    
    bars = 0
    
    digits = [i for i in a if i.isdigit()]
    money = int("".join(digits)) if digits else 0

    digits = [i for i in b if i.isdigit()]
    price = int("".join(digits)) if digits else 1
    
    if "-" in a or money < 0 or price <= 0:  
        return "Invalid Input"
    
    print(f"Money: {money}, Price per chocolate: {price}")
    
    while money >= price and money != 0:
        money -= price
        bars +=1
    
    eaten = 0
    x = 0
    while bars != 0:    
        bars -= 1
        eaten += 1
        x += 1
        if x == 3:
            bars+=1
            x = 0
    
    return(eaten)

print(count_eatable_chocolates("40$", "1$"))
