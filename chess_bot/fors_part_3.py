"""
Robust Binär Sökning / Intervallhalvering för optimering av schack-parametrar.
Designad för att förhindra parameterkollaps genom MAE-loss och unika minimigränser.
Nu med Multiprocessing!
"""

import os
import time
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from main import parse_fen
    from rating import evaluate_board
except ImportError:
    print("Varning: Kunde inte importera parse_fen/evaluate_board. Körs i fel mapp?")

# =====================================================================
# 1. OPTIMERINGS-INSTÄLLNINGAR
# =====================================================================
CSV_FILENAME = os.path.join("stockfish_self_play_results", "positions_ratings.csv")
SAMPLE_SIZE = ModuleNotFoundError              # Sätt till None för att köra på hela filen
MAX_ITERATIONS = 200000
MAX_MINUTES = 20                
OUTPUT_TXT = "optimized_perfect_search.txt"

RATE_PARAMS = {"OPEN_RATE", "END_RATE"}

MIN_BOUNDS = { 
    "KING_SAFETY_BONUS": 0.001,
    "CONTROL_CENTER_BONUS": 0.001,
    "BREAKING_PAWN_CHAINS_BONUS": 0.001,
    "bishop_pair_bonus": 0.001,
    "knight_on_the_rim_penalty": 0.001,
    "pawn_chain_bonus": 0.001,
    "PASSED_PAWN_BONUS": 0.001,
    "enemy_king_corner_bonus": 0.001,
    "enemy_king_center_bonus": 0.001,
    "hanging_piece_penalty": 0.001,     
    "squares_controlled_bonus": 0.001,
    "pieac_pos_bonus": 0.01,          
    "rook_open_file_bonus": 0.001,
    "isolated_pawn_penalty": 0.001,
    "OPEN_RATE": 0.0001,
    "END_RATE": 0.0001,
    "rook_on_seventh_rank_bonus": 0.5,

}

best_params = { 
    "BREAKING_PAWN_CHAINS_BONUS": 0.001,
    "CONTROL_CENTER_BONUS": 0.001,
    "END_RATE": 0.0001,
    "KING_SAFETY_BONUS": 0.0001,
    "OPEN_RATE": 0.0001,
    "PASSED_PAWN_BONUS": 0.001,
    "bishop_pair_bonus": 0.5,
    "enemy_king_center_bonus": 0.5,
    "enemy_king_corner_bonus": 1.0,
    "hanging_piece_penalty": 2.0,
    "isolated_pawn_penalty": 0.5,
    "knight_on_the_rim_penalty": 0.5,
    "pawn_chain_bonus": 2.0,
    "pieac_pos_bonus": 1.0,
    "rook_open_file_bonus": 0.5,
    "squares_controlled_bonus": 0.01,
    "rook_on_seventh_rank_bonus": 0.0005,
}

# =====================================================================
# 2. MULTIPROCESSING OCH UTVÄRDERING
# =====================================================================

def _eval_worker(args):
    """Hjälpfunktion för multiprocessing. Måste ligga på top-level."""
    item, weights = args
    board, current_color, state = item
    try:
        return evaluate_board(board, state, {}, current_color, weights)
    except Exception:
        return 0.0

def evaluate_all(parsed_boards, weights, pool):
    """Använder den inskickade process-poolen för att utvärdera bräden parallellt."""
    tasks = [(item, weights) for item in parsed_boards]
    results = pool.map(_eval_worker, tasks)
    return np.array(results)

def save_progress(param_keys, weights, mae, filename=OUTPUT_TXT):
    optimized_params = {}
    for i, key in enumerate(param_keys):
        optimized_params[key] = round(float(weights[i]), 4)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Perfekt Optimerade Parametrar (MAE={mae:.2f})\n")
        f.write("best_params = {\n")
        for key, val in optimized_params.items():
            f.write(f'    "{key}": {val},\n')
        f.write("}\n")

def sample_and_parse(full_df, sample_size, seed=None):
    # Om sample_size är None eller inte ett heltal, ta 100% av filen
    if sample_size is None or not isinstance(sample_size, int):
        df_sample = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        n = min(sample_size, len(full_df))
        df_sample = full_df.sample(n=n, random_state=seed).reset_index(drop=True)

    boards = []
    ok_mask = np.ones(len(df_sample), dtype=bool)
    fail_count = 0
    for i, fen in enumerate(df_sample["fen"].values):
        try:
            boards.append(parse_fen(fen))
        except Exception:
            boards.append(None)
            ok_mask[i] = False
            fail_count += 1

    if fail_count:
        print(f"    ({fail_count}/{len(df_sample)} FEN exkluderades)")

    valid_boards = [b for b, ok in zip(boards, ok_mask) if ok]
    valid_sf = df_sample["stockfish_eval"].values[ok_mask]
    return valid_boards, valid_sf

# =====================================================================
# 3. BINÄR SÖKNING
# =====================================================================

def binary_search_optimize(full_df, param_keys, initial_weights):
    current_weights = list(initial_weights)

    print("\n--- Startar Perfekt Sökning med Multiprocessing! ---")
    print("Laddar och fryser statiskt träningsurval...")
    
    # DATAN FRYSES HÄR: Laddas enbart EN gång före loopen med fast seed!
    parsed_boards, sf_evals = sample_and_parse(full_df, SAMPLE_SIZE, seed=42)
    
    # Startar process-poolen en gång för hela sökningen
    num_cores = mp.cpu_count() - 1
    print(f"Använder {num_cores} CPU-kärnor för utvärdering.")
    
    with mp.Pool(processes=num_cores) as pool:
        initial_dict = {param_keys[i]: current_weights[i] for i in range(len(param_keys))}
        preds_current = evaluate_all(parsed_boards, initial_dict, pool)
        
        best_mae = np.mean(np.abs(preds_current - sf_evals))
        print(f"Initial MAE på det fasta urvalet: {best_mae:.4f}")
        
        start_time = time.time()
        time_limit_seconds = MAX_MINUTES * 60

        for it in range(MAX_ITERATIONS):
            if time.time() - start_time >= time_limit_seconds:
                print(f"\n*** Tidsgränsen på {MAX_MINUTES} minuter nådd! ***")
                break

            print(f"\n>>> ITERATION {it + 1}/{MAX_ITERATIONS} <<<")
            print(f"    MAE vid start av runda: {best_mae:.4f}")

            improved_this_iteration = False
            
            shuffled_indices = list(range(len(param_keys)))
            random.shuffle(shuffled_indices)

            for idx in tqdm(shuffled_indices, desc=f"Runda {it + 1}", colour="green"):
                if time.time() - start_time >= time_limit_seconds:
                    return current_weights, best_mae

                key = param_keys[idx]
                current_val = current_weights[idx]
                best_val_for_key = current_val

                step = 1.0              
                max_step = 16.0         
                tolerance_step = 0.001
                min_allowed = MIN_BOUNDS.get(key, 0.001)

                while step >= tolerance_step:
                    base_val = current_val if abs(current_val) >= 1e-4 else min_allowed

                    test_vals = [
                        base_val * (1.0 - step),
                        base_val * (1.0 + step)
                    ]

                    if key in RATE_PARAMS:
                        test_vals = [v for v in test_vals if min_allowed <= v <= 0.99]
                    else:
                        test_vals = [v for v in test_vals if v >= min_allowed]

                    improved_in_step = False
                    for val in test_vals:
                        if val == current_val:
                            continue

                        test_weights = list(current_weights)
                        test_weights[idx] = val
                        weight_dict = {param_keys[j]: test_weights[j] for j in range(len(param_keys))}

                        preds = evaluate_all(parsed_boards, weight_dict, pool)
                        if np.array_equal(preds, preds_current):
                            continue

                        mae = np.mean(np.abs(preds - sf_evals))

                        if best_mae - mae > 0.0001:
                            print(f"  [+] {key} förbättrades! {current_val:.4f} -> {val:.4f} (Ny MAE: {mae:.4f})")
                            best_mae = mae
                            best_val_for_key = val
                            improved_this_iteration = True
                            improved_in_step = True
                            
                            # Cachen uppdateras för att tidiga hopp ska fungera!
                            preds_current = preds

                    if improved_in_step:
                        current_val = best_val_for_key
                        step = min(step * 1.5, max_step)
                    else:
                        step /= 2.0

                current_weights[idx] = best_val_for_key

            save_progress(param_keys, current_weights, best_mae)

            if not improved_this_iteration:
                print("    Ingen förbättring denna runda (konvergens nådd).")
                break

    return current_weights, best_mae

def main():
    if not os.path.exists(CSV_FILENAME):
        print(f"Kunde inte hitta {CSV_FILENAME}.")
        return

    full_df = pd.read_csv(CSV_FILENAME)
    full_df = full_df.rename(columns={"eval_cp": "stockfish_eval"})
    
    # Clip och konvertering till Bönder från Centipawns
    full_df["stockfish_eval"] = full_df["stockfish_eval"].clip(-1500, 1500) / 100.0  
    
    param_keys = sorted(best_params.keys())
    initial_weights = [float(best_params[k]) for k in param_keys]

    t0 = time.time()
    w, final_mae = binary_search_optimize(full_df, param_keys, initial_weights)

    print(f"\nBinär sökning avslutades efter {time.time() - t0:.2f} sekunder. Slutlig MAE: {final_mae:.2f}")

if __name__ == "__main__":
    main()