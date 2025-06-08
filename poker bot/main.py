
from poker_bot import parse_card, parallel_multi_player_simulation

def main():
    print("=== Texas Hold'em Oddsanalys (Svensk notation) ===")
    print("Format: hE (hjärter ess), sKn (spader knekt), r7 (ruter 7), kD (klöver dam)")
    print("Tillåtna valörer: 2-9, 10, Kn (knekt), D (dam), K (kung), E (ess)")
    print("Skriv '-' eller tryck Enter för att ändra föregående kort.")
    print("Skriv 'fold' för att lägga dig och börja om med en ny hand.")
    print("Skriv 'q' eller 'quit' för att avsluta spelet.\n")

    while True:
        try:
            num_players = int(input("Ange antal spelare (inkl dig, 2–10): ").strip())
            if not (2 <= num_players <= 10):
                print("Antal spelare måste vara mellan 2 och 10.")
                continue
        except ValueError:
            print("Ogiltigt tal.")
            continue

        simulations = 1000_00
        processes = 3

        player_hand = []
        card_index = 0
        while card_index < 2:
            print(f"\nValörer: 2–9, 10, Kn (11), D (12), K (13), E (14)")
            print("Färger: h = Hjärter, s = Spader, r = Ruter, k = Klöver")
            card_str = input(f"Spelarkort {card_index + 1}: ").strip().lower()

            if card_str in ["fold"]:
                print("Du har lagt dig (fold). Nästa hand börjar.\n")
                break
            elif card_str in ["", "-"]:
                if card_index > 0:
                    print("Återgår ett steg...")
                    card_index -= 1
                    player_hand.pop()
                else:
                    print("Inget att ångra ännu.")
                continue
            elif card_str in ["q", "quit"]:
                print("Avslutar spelet.")
                return
            try:
                card = parse_card(card_str)
                if card in player_hand:
                    print("Du har redan det kortet.")
                    continue
                player_hand.append(card)
                card_index += 1
            except Exception:
                print("Ogiltigt kort. Försök igen.")

        if len(player_hand) < 2:
            continue

        board = []

        print("\n--- Preflopanalys ---")
        win, tie = parallel_multi_player_simulation(player_hand, board, num_players, simulations=3000, processes=processes)
        print(f"Vinstchans: {win:.2f}%, Oavgjort: {tie:.2f}%")

        for phase, target_len, label in zip(["flop", "turn", "river"], [3, 4, 5], ["Flopkort", "Turnkort", "Riverkort"]):
            print(f"\nMata in {label.lower()}:")
            while len(board) < target_len:
                print(f"\nValörer: 2–9, 10, Kn (11), D (12), K (13), E (14)")
                print("Färger: h = Hjärter, s = Spader, r = Ruter, k = Klöver")
                card_str = input(f"{label} {len(board)+1 - (3 if phase=='flop' else 4)}: ").strip().lower()
                if card_str in ["fold"]:
                    print("Du har lagt dig (fold). Nästa hand börjar.\n")
                    break
                elif card_str in ["", "-"]:
                    if len(board) > target_len - (3 if phase=='flop' else 4):
                        print("Återgår ett steg...")
                        board.pop()
                    else:
                        print("Inget att ångra ännu.")
                    continue
                elif card_str in ["q", "quit"]:
                    print("Avslutar spelet.")
                    return
                try:
                    card = parse_card(card_str)
                    if card in player_hand or card in board:
                        print("Kortet har redan använts.")
                        continue
                    board.append(card)
                except:
                    print("Fel format. Försök igen.")

            if len(board) < target_len:
                break

            print(f"\n--- {phase.capitalize()}analys ---")
            win, tie = parallel_multi_player_simulation(player_hand, board, num_players, simulations=simulations, processes=processes)
            print(f"Vinstchans: {win:.2f}%, Oavgjort: {tie:.2f}%")

        print("\n--- Ny hand? ---")
        svar = input("Tryck Enter för ny hand eller 'q' för att avsluta: ").strip().lower()
        if svar in ["q", "quit"]:
            print("Tack för att du spelade!")
            break

if __name__ == '__main__':
    main()
