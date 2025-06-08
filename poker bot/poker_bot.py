
import random
from collections import Counter
from itertools import combinations

# Färger (suits) och valörer (ranks) på svenska
SUITS = {'h': 'Hjärter', 's': 'Spader', 'r': 'Ruter', 'k': 'Klöver'}
RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, 't': 10,
    'kn': 11, 'd': 12, 'k': 13, 'e': 14
}

def parse_card(card_str):
    card_str = card_str.lower().strip()
    if len(card_str) < 2:
        raise ValueError("Kortsträng för kort.")
    suit = card_str[0]
    rank_part = card_str[1:]
    if rank_part == 'kn':
        rank = 'kn'
    elif rank_part in RANKS:
        rank = rank_part
    elif rank_part == 't':
        rank = '10'
    else:
        raise ValueError("Ogiltig valör.")
    return (suit, RANKS[rank])

def generate_deck(exclude_cards):
    deck = [(s, r) for s in SUITS for r in RANKS.values()]
    return [card for card in deck if card not in exclude_cards]

def check_straight(values):
    values = sorted(set(values), reverse=True)
    for i in range(len(values) - 4):
        slice_ = values[i:i+5]
        if slice_[0] - slice_[-1] == 4:
            return True, slice_[0]
    if set([14, 2, 3, 4, 5]).issubset(set(values)):
        return True, 5
    return False, None

def hand_rank(hand):
    values = sorted([r for _, r in hand], reverse=True)
    suits = [s for s, _ in hand]
    value_counter = Counter(values)
    suit_counter = Counter(suits)

    is_flush = max(suit_counter.values()) >= 5
    is_straight, straight_high = check_straight(values)

    if is_flush:
        flush_suit = suit_counter.most_common(1)[0][0]
        flush_cards = sorted([r for s, r in hand if s == flush_suit], reverse=True)
        is_straight_flush, sf_high = check_straight(flush_cards)
        if is_straight_flush:
            return (8, sf_high)

    if 4 in value_counter.values():
        four = [val for val, count in value_counter.items() if count == 4][0]
        kicker = max([v for v in values if v != four])
        return (7, four, kicker)

    if sorted(value_counter.values(), reverse=True)[:2] == [3, 2]:
        three = [val for val, count in value_counter.items() if count == 3][0]
        pair = [val for val, count in value_counter.items() if count == 2][0]
        return (6, three, pair)

    if is_flush:
        flush_cards = sorted([r for s, r in hand if s == flush_suit], reverse=True)
        return (5, flush_cards[:5])

    if is_straight:
        return (4, straight_high)

    if 3 in value_counter.values():
        three = [val for val, count in value_counter.items() if count == 3][0]
        kickers = [v for v in values if v != three][:2]
        return (3, three, kickers)

    pairs = [val for val, count in value_counter.items() if count == 2]
    if len(pairs) >= 2:
        top2 = sorted(pairs, reverse=True)[:2]
        kicker = max([v for v in values if v not in top2])
        return (2, top2, kicker)

    if len(pairs) == 1:
        pair = pairs[0]
        kickers = [v for v in values if v != pair][:3]
        return (1, pair, kickers)

    return (0, values[:5])

def compare_hands(hand1, hand2):
    return (hand1 > hand2) - (hand1 < hand2)

def monte_carlo_simulation(player_hand, board, simulations=10000):
    wins, ties = 0, 0
    deck = generate_deck(player_hand + board)
    for _ in range(simulations):
        deck_copy = deck[:]
        random.shuffle(deck_copy)
        opp_hand = [deck_copy.pop(), deck_copy.pop()]
        remaining_board = board[:]
        while len(remaining_board) < 5:
            remaining_board.append(deck_copy.pop())
        player_best = best_hand(player_hand + remaining_board)
        opp_best = best_hand(opp_hand + remaining_board)
        result = compare_hands(player_best, opp_best)
        if result == 1:
            wins += 1
        elif result == 0:
            ties += 1
    return 100 * wins / simulations, 100 * ties / simulations

def best_hand(seven_cards):
    return max((hand_rank(list(c)) for c in combinations(seven_cards, 5)))


