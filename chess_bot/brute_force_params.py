"""
Bruteforce / Coordinate Descent Optimering av schack-parametrar.
Testar att ändra en parameter i taget baserat på fasta procentuella steg.
Stoppar automatiskt efter en angiven tidsgräns i minuter.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Försök importera från din motor (enbart funktionerna, inga variabler)
try:
    from main import parse_fen
    from rating import evaluate_board
except ImportError:
    print("Varning: Kunde inte importera parse_fen/evaluate_board. Körs i fel mapp?")


# =====================================================================
# 1. OPTIMERINGS-INSTÄLLNINGAR (Kontrollrummet)
# =====================================================================
CSV_FILENAME = "evaluation_comparison.csv"
SAMPLE_SIZE = 1000         # Hur många bräden vi utvärderar per test
MAX_ITERATIONS = 200000        # Max antal varv om tiden inte tar slut först
MAX_MINUTES = 60 * 16          # <- NYTT: Stoppa efter så här många minuter (t.ex. 5.0 eller 0.5 för 30 sek)
# Lista på procentuella ändringar att testa per parameter.
PERCENTAGE_STEPS = [
    -0.50, -0.25, -0.10, -0.05, 
    0.05, 0.10, 0.25, 0.50, -0.9, 2.0, -0.3, 0.3, -0.2, 0.2, -0.15, 0.15
]

# Om en parameter är exakt 0 från start fungerar inte procent. 
ZERO_FALLBACK_STEPS = [
    -2.0, -1.0, -0.5, -0.1, 
    0.1, 0.5, 1.0, 2.0
]

# Parametrar som är procentsatser och INTE ska multipliceras med scale på slutet
RATE_PARAMS = {"OPEN_RATE", "END_RATE"}

# =====================================================================
# 2. START-PARAMETRAR
# =====================================================================
best_params = {
    "BREAKING_PAWN_CHAINS_BONUS": 0.01,
    "CONTROL_CENTER_BONUS": 3.3876,
    "END_RATE": 0.3995,
    "KING_SAFETY_BONUS": 7.0014,
    "OPEN_RATE": 0.4542,
    "PASSED_PAWN_BONUS": 0.8247,
    "bishop_pair_bonus": 2.5685,
    "enemy_king_center_bonus": 1.1242,
    "enemy_king_corner_bonus": 1.0072,
    "hanging_piece_penalty": 2.3199,
    "knight_on_the_rim_penalty": 2.7241,
    "pawn_chain_bonus": 0.01,
    "pieac_pos_bonus": 0.0588,
    "squares_controlled_bonus": 0.0579,
}
# =====================================================================


def load_data(csv_filename, sample_size, seed=42):
    df = pd.read_csv(csv_filename)
    fixed_columns = {"game", "move", "turn", "our_eval", "stockfish_eval", "fen"}
    param_keys = [c for c in df.columns if c not in fixed_columns]
    
    df["stockfish_eval"] = df["stockfish_eval"].clip(-2000, 2000)
    n = min(sample_size, len(df))
    df_sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df_sample, param_keys


def parse_positions(fens):
    parsed = []
    for fen in tqdm(fens, desc="Parsar FEN-strängar", colour="cyan"):
        try:
            parsed.append(parse_fen(fen))
        except Exception:
            parsed.append(None)
    return parsed


def evaluate_single(item, weights):
    if item is None:
        return 0.0
    board, current_color, state = item
    try:
        return evaluate_board(board, state, {}, current_color, weights)
    except Exception:
        return 0.0


def evaluate_all(parsed_boards, weights):
    return np.array([evaluate_single(item, weights) for item in parsed_boards])


def best_scale(preds, targets):
    denom = np.dot(preds, preds)
    if denom == 0:
        return 1.0
    scale = np.dot(preds, targets) / denom
    return float(np.clip(scale, 0.01, 1000.0))


def bruteforce_optimize(parsed_boards, param_keys, sf_evals, initial_weights):
    current_weights = list(initial_weights)
    
    initial_dict = {param_keys[i]: current_weights[i] for i in range(len(param_keys))}
    preds = evaluate_all(parsed_boards, initial_dict)
    best_scale_val = best_scale(preds, sf_evals)
    best_mse = np.mean((best_scale_val * preds - sf_evals) ** 2)
    
    print(f"\n--- Startar Bruteforce! Initial MSE: {best_mse:.2f} ---")
    print(f"Tidsgräns satt till: {MAX_MINUTES} minuter.")

    start_time = time.time()
    time_limit_seconds = MAX_MINUTES * 60

    for it in range(MAX_ITERATIONS):
        # Kolla om tiden har runnit ut innan vi börjar en ny iteration
        if time.time() - start_time >= time_limit_seconds:
            print(f"\n*** Tidsgränsen på {MAX_MINUTES} minuter har nåtts! Avbryter sökningen. ***")
            break

        print(f"\n>>> ITERATION {it+1}/{MAX_ITERATIONS} <<<")
        improved_this_iteration = False

        for i, key in enumerate(param_keys):
            # Kolla tiden även inuti loopen så den bryter snabbt när tiden är ute
            if time.time() - start_time >= time_limit_seconds:
                print(f"\n*** Tidsgränsen på {MAX_MINUTES} minuter har nåtts! Avbryter sökningen. ***")
                return current_weights, best_scale_val, best_mse

            test_start = time.time()
            current_val = current_weights[i]
            
            if current_val == 0:
                test_vals = ZERO_FALLBACK_STEPS
            else:
                test_vals = [current_val * (1 + pct) for pct in PERCENTAGE_STEPS]

            # Om det är en rate-parameter, håll den inom rimliga gränser
            if key in RATE_PARAMS:
                test_vals = [v for v in test_vals if 0.01 <= v <= 0.99]
            
            best_val_for_key = current_val
            
            for val in test_vals:
                if val == current_val:
                    continue
                    
                test_weights = list(current_weights)
                test_weights[i] = val
                weight_dict = {param_keys[j]: test_weights[j] for j in range(len(param_keys))}
                
                preds = evaluate_all(parsed_boards, weight_dict)
                scale = best_scale(preds, sf_evals)
                mse = np.mean((scale * preds - sf_evals) ** 2)
                
                if mse < best_mse:
                    print(f"  [+] {key} förbättrades! {current_val:.4f} -> {val:.4f} (Ny MSE: {mse:.2f})")
                    best_mse = mse
                    best_val_for_key = val
                    best_scale_val = scale
                    improved_this_iteration = True
            
            test_end = time.time()
            print(f"    Testade {key} på {test_end - test_start:.2f} sekunder. Bästa MSE hittills: {best_mse:.2f}")
            
            current_weights[i] = best_val_for_key

        if not improved_this_iteration:
            pass
            
    return current_weights, best_scale_val, best_mse


def main():
    if not os.path.exists(CSV_FILENAME):
        print(f"Kunde inte hitta {CSV_FILENAME}.")
        return

    df_sample, param_keys = load_data(CSV_FILENAME, SAMPLE_SIZE)
    print(f"Laddade {len(df_sample)} positioner, {len(param_keys)} parametrar.")

    train_boards = parse_positions(df_sample["fen"].values)
    train_sf = df_sample["stockfish_eval"].values

    try:
        initial_weights = [float(best_params[k]) for k in param_keys]
        print("Laddade startvikter från globala 'best_params'.")
    except KeyError as e:
        print(f"Varning: Nyckeln {e} saknas i 'best_params'. Fallback till CSV-rad.")
        initial_weights = [float(df_sample.iloc[-1][k]) for k in param_keys]

    t0 = time.time()
    w, scale, final_mse = bruteforce_optimize(train_boards, param_keys, train_sf, initial_weights)
    
    print(f"\nBruteforce avslutades efter {time.time() - t0:.2f} sekunder.")

    optimized_params = {}
    for i, key in enumerate(param_keys):
        if key in RATE_PARAMS:
            optimized_params[key] = round(float(w[i]), 4)
        else:
            optimized_params[key] = round(float(w[i] * scale), 4)

    print("\nNya optimerade parametrar:")
    for key, val in optimized_params.items():
        print(f'    "{key}": {val},')

    with open("optimized_bruteforce.txt", "w", encoding="utf-8") as f:
        f.write("# Bruteforce optimerade parametrar\n")
        f.write("best_params = {\n")
        for key, val in optimized_params.items():
            f.write(f'    "{key}": {val},\n')
        f.write("}\n")
    print("\nSparat till 'optimized_bruteforce.txt'!")

if __name__ == "__main__":
    main()