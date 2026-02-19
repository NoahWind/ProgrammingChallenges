
from multiprocessing import Pool
import random
from collections import Counter
from itertools import combinations

SUITS = {'h': 'Hjärter', 's': 'Spader', 'r': 'Ruter', 'k': 'Klöver'}
RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, 't': 10,
    'kn': 11, 'd': 12, 'k': 13, 'e': 14
}

def parse_card(card_str):
    if isinstance(card_str, tuple):
        return card_str
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
    return [(s, r) for s in SUITS for r in RANKS.values() if (s, r) not in exclude_cards]

def check_straight(values):
    values = sorted(set(values), reverse=True)
    for i in range(len(values) - 4):
        if values[i] - values[i+4] == 4:
            return True, values[i]
    if set([14, 2, 3, 4, 5]).issubset(values):
        return True, 5
    return False, None

def hand_rank(hand):
    values = sorted([r for _, r in hand], reverse=True)
    suits = [s for s, _ in hand]
    vc = Counter(values)
    sc = Counter(suits)

    flush = None
    for suit, count in sc.items():
        if count >= 5:
            flush = suit
            break

    if flush:
        flush_values = sorted([r for s, r in hand if s == flush], reverse=True)
        is_sf, sf_high = check_straight(flush_values)
        if is_sf:
            return (8, sf_high)

    freq = sorted(vc.items(), key=lambda x: (-x[1], -x[0]))
    counts = [x[1] for x in freq]
    vals = [x[0] for x in freq]

    if counts[0] == 4:
        return (7, vals[0], vals[1])
    if counts[0] == 3 and counts[1] >= 2:
        return (6, vals[0], vals[1])
    if flush:
        top = sorted([r for s, r in hand if s == flush], reverse=True)[:5]
        return (5, top)
    is_straight, high = check_straight(values)
    if is_straight:
        return (4, high)
    if counts[0] == 3:
        return (3, vals[0], vals[1:3])
    if counts[0] == 2 and counts[1] == 2:
        return (2, vals[:2], vals[2])
    if counts[0] == 2:
        return (1, vals[0], vals[1:4])
    return (0, vals[:5])

def best_hand(seven_cards):
    return max((hand_rank(c) for c in combinations(seven_cards, 5)))

def compare_hands(h1, h2):
    return (h1 > h2) - (h1 < h2)

def monte_carlo_worker_fast(args):
    hand, board, deck, num_opponents = args
    random.shuffle(deck)
    opps = [[deck.pop(), deck.pop()] for _ in range(num_opponents)]
    full_board = board[:]
    while len(full_board) < 5:
        full_board.append(deck.pop())
    player = best_hand(hand + full_board)
    result = 1
    for opp in opps:
        r = compare_hands(player, best_hand(opp + full_board))
        if r < 0:
            return -1
        elif r == 0:
            result = 0
    return result

def parallel_multi_player_simulation_fast(player_hand, board, num_players=5, simulations=10000, processes=6):
    num_opponents = num_players - 1
    deck = generate_deck(player_hand + board)
    args = [(player_hand, board, deck[:], num_opponents) for _ in range(simulations)]
    with Pool(processes=processes) as pool:
        results = pool.map(monte_carlo_worker_fast, args)
    wins = sum(1 for r in results if r == 1)
    ties = sum(1 for r in results if r == 0)
    return 100 * wins / simulations, 100 * ties / simulations

def parallel_multi_player_simulation(player_hand, board, num_players=5, simulations=10000, processes=6):
    encoded_hand = [parse_card(c) for c in player_hand]
    encoded_board = [parse_card(c) for c in board]
    return parallel_multi_player_simulation_fast(encoded_hand, encoded_board, num_players, simulations, processes)
