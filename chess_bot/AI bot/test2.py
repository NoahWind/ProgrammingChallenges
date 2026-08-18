import chess.polyglot

# Eftersom den nu ligger i samma mapp använder vi bara filnamnet
BOOK_PATH = "gm2001.bin"

def count_total_entries(path):
    print(f"Räknar ställningar i {path}...")
    try:
        with chess.polyglot.open_reader(path) as reader:
            # Vi loopar igenom hela boken och räknar varje post
            total = sum(1 for _ in reader)
            print(f"----------------------------------------")
            print(f"Totalt antal poster i boken: {total:,}")
            print(f"----------------------------------------")
    except Exception as e:
        print(f"Kunde inte läsa filen: {e}")

if __name__ == "__main__":
    count_total_entries(BOOK_PATH)