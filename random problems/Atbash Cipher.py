"""  
Encodes a word by replacing each letter with its **reverse counterpart** in the alphabet.  
- 'a' ↔ 'z', 'b' ↔ 'y', ..., 'z' ↔ 'a'  
- 'A' ↔ 'Z', 'B' ↔ 'Y', ..., 'Z' ↔ 'A'  
- Non-alphabetic characters remain unchanged.  
"""

word =list(input(""))

def swap(word):
    alphabet = [chr(i) for i in range(97, 123)]
    capital_alphabet = [chr(i) for i in range(65, 91)]
    
    new_word = []
    
    for letter in word:
        if letter in alphabet:
            pos = alphabet.index(letter)
            new_word.append(alphabet[len(alphabet) - pos - 1])
        elif letter in capital_alphabet:
            pos = capital_alphabet.index(letter)
            new_word.append(capital_alphabet[len(capital_alphabet) - pos - 1])
        else:
            new_word.append(letter)
    
    return new_word

print("".join(swap(word)))