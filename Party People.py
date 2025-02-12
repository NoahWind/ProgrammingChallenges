"""
A party simulation where attendees leave based on their minimum required attendance.
If too many people leave, others may follow, triggering a chain reaction.
The process repeats recursively until only those who are willing to stay remain.
The goal: count how many people stay.
"""

prefrenses = [int(x) for x in input().replace(',', ' ').split()]

def filter(prefrenses):
    filterd = False
    new_prefremses = []
    amount = len(prefrenses)
    for i in prefrenses:
        if i <= amount:
            new_prefremses.append(i)
        else:
            filterd = True
    
    if filterd:
        return(filter(new_prefremses))
    else:
        return len(new_prefremses)

print(filter(prefrenses))
    