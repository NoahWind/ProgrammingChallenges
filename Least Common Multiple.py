b = input().replace(",", " ").split()

def make_prime(b):
    b_prime = list(map(int, b))
    for i in b_prime[:]:
        for j in range(2, i):
            if i == 1:
                continue
            if i % j == 0:
                b_prime.append(j)
                b_prime.append(i // j)
                b_prime.remove(i)
                return make_prime(b_prime)
    return b_prime

def get_largest_occurense(b_prime):
    max_counts = {}
    current_count = 1

    for i in range(1, len(b_prime)):
        if b_prime[i] == b_prime[i - 1]:
            current_count += 1
        else:
            num = b_prime[i - 1]
            max_counts[num] = max(max_counts.get(num, 0), current_count) 
            current_count = 1

    num = b_prime[-1]
    max_counts[num] = max(max_counts.get(num, 0), current_count)
    
    new_b_prime = []
    
    for i in max_counts:
        new_b_prime.extend([i] * max_counts[i])

    return new_b_prime
                
def lcm(b):
    b_prime = make_prime(b)
    b_prime = get_largest_occurense(b_prime)
    b_prime.sort()
    b = 1
    for i in b_prime:
        b *= i
    return b
        
b = lcm(b)
print(b)
