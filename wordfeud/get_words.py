import requests
import os

def hamta_text_fran_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.content.decode('utf-8', errors='ignore')
    except:
        return ""

def maximera_ordlista():
    ord_set = set()
    existerande = ['wordfeud_lista.txt', 'wordfeud_superlista.txt']
    
    for f_namn in existerande:
        if os.path.exists(f_namn):
            with open(f_namn, 'r', encoding='utf-8') as f:
                for rad in f:
                    ord_set.add(rad.strip().lower())

    url = "https://raw.githubusercontent.com/titoBouzout/Dictionaries/master/Swedish.dic"
    data = hamta_text_fran_url(url)
    
    for rad in data.splitlines():
        o = rad.split('/')[0].strip().lower()
        if 2 <= len(o) <= 15 and o.isalpha():
            ord_set.add(o)

    sorterat = sorted(list(ord_set))
    with open("wordfeud_master.txt", "w", encoding="utf-8") as f:
        for ordet in sorterat:
            f.write(ordet + "\n")

    print(f"Antal ord: {len(sorterat)}")

if __name__ == "__main__":
    maximera_ordlista()