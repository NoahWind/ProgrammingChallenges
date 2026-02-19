import os

sökväg = r"C:\Users\Noah\Documents\code"

for rot, _, filer in os.walk(sökväg):
    for fil in filer:
        if "seek" in fil.lower():
            print(os.path.join(rot, fil))
