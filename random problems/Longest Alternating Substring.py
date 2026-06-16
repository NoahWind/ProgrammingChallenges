"""  
Finds the longest subsequence of alternating even and odd digits in a given number string.  
"""
sub_string = "721449827599186159274227324466"

def odd_or_even(digit):
    return digit % 2 == 0

def longest_alt(sub_string):
    sub_list = [int(digit) for digit in sub_string]
    
    longest = []
    current = [sub_list[0]]
    
    for i in range(1, len(sub_list)):
        if odd_or_even(sub_list[i]) != odd_or_even(sub_list[i - 1]):
            current.append(sub_list[i])
            
        else:
            if len(current) > len(longest):
                longest = current[:]
                
            current = [sub_list[i]]

    if len(current) > len(longest):
        longest = current

    return longest
        
print(longest_alt(sub_string))
