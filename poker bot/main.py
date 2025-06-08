from poker_bot import parse_card, monte_carlo_simulation

def main():
    print("=== Texas Hold'em Oddsanalys (Svensk notation) ===")
    print("Format: hE (hjärter ess), sKn (spader knekt), r7 (ruter 7), kD (klöver dam)")
    print("Tillåtna valörer: 2-9, 10, Kn (knekt), D (dam), K (kung), E (ess)")
    print("Skriv '-' eller tryck Enter för att ändra föregående kort.")
    print("Skriv 'fold' för att lägga dig och börja om med en ny hand.")
    print("Skriv 'q' eller 'quit' för att avsluta spelet.\n")

    while True:
        # Fråga om simulationshastighet
        while True:
            try:
                speed = input("Ange antal simuleringar (1,2,3,...): ").strip()
                if speed.lower() in ["q", "quit"]:
                    print("Avslutar spelet.")
                    return
                speed = int(speed)
                if speed <= 0:
                    print("Ange ett positivt heltal.")
                    continue
                speed *= 1000
                break
            except ValueError:
                print("Ogiltig inmatning. Ange ett heltal.")

        # === Spelarens hand ===
        player_hand = []
        card_index = 0
        while card_index < 2:
            print(f"\nValörer: 2–9, 10, Kn (11), D (12), K (13), E (14)")
            print("Färger: h = Hjärter, s = Spader, r = Ruter, k = Klöver")
            card_str = input(f"Spelarkort {card_index + 1}: ").strip().lower()

            if card_str in ["fold"]:
                print("Du har lagt dig (fold). Nästa hand börjar.\n")
                break  # gå direkt till nästa iteration av while True
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
            continue  # hand ej komplett, börja om

        board = []

        print("\n--- Preflopanalys ---")
        win, tie = monte_carlo_simulation(player_hand, board, simulations=3000)
        print(f"Vinstchans: {win:.2f}%, Oavgjort: {tie:.2f}%")

        print("\nMata in 3 kort till floppen:")
        while len(board) < 3:
            print(f"\nValörer: 2–9, 10, Kn (11), D (12), K (13), E (14)")
            print("Färger: h = Hjärter, s = Spader, r = Ruter, k = Klöver")
            card_str = input(f"Flopkort {len(board)+1}: ").strip().lower()
            if card_str in ["fold"]:
                print("Du har lagt dig (fold). Nästa hand börjar.\n")
                break
            elif card_str in ["", "-"]:
                if board:
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

        if len(board) < 3:
            continue

        print("\n--- Flopanalys ---")
        win, tie = monte_carlo_simulation(player_hand, board, simulations=speed)
        print(f"Vinstchans: {win:.2f}%, Oavgjort: {tie:.2f}%")

        print("\nMata in turnkort:")
        while len(board) < 4:
            print(f"\nValörer: 2–9, 10, Kn (11), D (12), K (13), E (14)")
            print("Färger: h = Hjärter, s = Spader, r = Ruter, k = Klöver")
            card_str = input("Turnkort: ").strip().lower()
            if card_str in ["fold"]:
                print("Du har lagt dig (fold). Nästa hand börjar.\n")
                break
            elif card_str in ["", "-"]:
                if len(board) > 3:
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

        if len(board) < 4:
            continue

        print("\n--- Turnanalys ---")
        win, tie = monte_carlo_simulation(player_hand, board, simulations=speed)
        print(f"Vinstchans: {win:.2f}%, Oavgjort: {tie:.2f}%")

        print("\nMata in riverkort:")
        while len(board) < 5:
            print(f"\nValörer: 2–9, 10, Kn (11), D (12), K (13), E (14)")
            print("Färger: h = Hjärter, s = Spader, r = Ruter, k = Klöver")
            card_str = input("Riverkort: ").strip().lower()
            if card_str in ["fold"]:
                print("Du har lagt dig (fold). Nästa hand börjar.\n")
                break
            elif card_str in ["", "-"]:
                if len(board) > 4:
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

        if len(board) < 5:
            continue

        print("\n--- Slutgiltig analys (river) ---")
        win, tie = monte_carlo_simulation(player_hand, board, simulations=speed)
        print(f"Slutlig vinstchans: {win:.2f}%, Oavgjort: {tie:.2f}%")

        print("\n--- Ny hand? ---")
        svar = input("Tryck Enter för ny hand eller 'q' för att avsluta: ").strip().lower()
        if svar in ["q", "quit"]:
            print("Tack för att du spelade!")
            break


if __name__ == '__main__':
    main()
