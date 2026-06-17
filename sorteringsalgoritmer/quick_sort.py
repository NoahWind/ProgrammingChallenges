import random

LEANGHT = 100000
arr = []

for i in range(LEANGHT):
    arr.append(random.randint(1,99999))
print(arr)

def is_sorted(arr):
    a = None

    for obj in arr:
        if a == None:
            a = obj
        elif a>obj:
            return False
        a = obj
    return True

def quick_sort(arr):
    if len(arr)==0:
        return arr
    list_high = []
    pivot_list = []
    list_low = []

    pivot = random.choice(arr)
    for obj in arr:
        if obj > pivot:
            list_high.append(obj)
        elif obj < pivot:
            list_low.append(obj)
        else:
            pivot_list.append(obj)
    sum = quick_sort(list_low) + pivot_list +  quick_sort(list_high) # Herregud
    return sum

print(is_sorted(quick_sort(arr)))