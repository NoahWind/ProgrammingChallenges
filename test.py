import random
import multiprocessing

def simulate_blackjack(seed, big_max):
    random.seed(seed)  # Unik seed för varje process
    start_am = 1
    current = start_am
    max_amount = 0
    x = 0
    odds = 0.5  

    while True:
        me = random.randint(0, 1)
        house = random.randint(0, 1)
        
        x += 1

        if house == me:  # Vinst: Dubbla pengarna
            current *= 2
            odds *= 0.5
        else:  # Förlust: Återställ pengar
            if max_amount < current:
                max_amount = current

                with big_max.get_lock():  # Lås så att uppdateringar sker korrekt mellan processer
                    if big_max.value < max_amount:
                        big_max.value = max_amount
                        print(f"\n[Process {seed}] Försök: {x} | Max vinst: {max_amount} | Odds: {odds}")

            current = start_am
            odds = 0.5  # Återställ oddsen

def run_parallel_simulations(n_processes):
    with multiprocessing.Manager() as manager:
        big_max = manager.Value('i', 0)  # Delad variabel mellan processerna
        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(simulate_blackjack, [(i, big_max) for i in range(n_processes)])

        print(f"\n=== Global Maxvinst efter {n_processes} simuleringar: {big_max.value} ===")

if __name__ == "__main__":
    num_simulations = 6  # Antal parallella simuleringar
    run_parallel_simulations(num_simulations)
