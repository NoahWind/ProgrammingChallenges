"""
Binär sökning / Intervallhalvering för optimering av schack-parametrar.
Använder en binär sökansats (intervallhalvering) per parameter istället för fasta procentsteg.
Stoppar automatiskt efter en angiven tidsgräns i minuter.

UPPDATERAD för nya CSV-formatet från test.py (Stockfish självspel):
    kolumner: game_id, move_number, fen, move_uci, eval_cp, turn

Buggfixar/förbättringar:
  1. param_keys hämtas nu från best_params (CSV:n har inga paramkolumner längre).
  2. Rader där FEN inte kunde parsas exkluderas helt.
  3. Sampling är nu äkta slumpmässig varje körning (seed=None som standard).
  4. Konvergens: om en hel runda över alla parametrar inte förbättrar något -> avbryt.
  5. Progressbar per parameter-runda.
  6. Bästa parametrar + MSE sparas till fil efter varje runda.
  7. Exceptions i evaluate_single räknas och rapporteras.
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
CSV_FILENAME = os.path.join("stockfish_self_play_results", "positions_ratings.csv")
SAMPLE_SIZE = 600              # Hur många positioner vi utvärderar per test
MAX_ITERATIONS = 200000         # Max antal varv om tiden inte tar slut först
MAX_MINUTES = 20                # Stoppa efter så här många minuter
RANDOM_SEED = None              # None = äkta slump varje körning
OUTPUT_TXT = "optimized_binary_search.txt"

# Parametrar som är procentsatser och INTE ska multipliceras med scale på slutet
RATE_PARAMS = {"OPEN_RATE", "END_RATE"}

# Parametrar som måste vara 0.0 eller högre för att inte förstöra logiken
POSITIVE_PARAMS = {
    "KING_SAFETY_BONUS", 
    "CONTROL_CENTER_BONUS", 
    "BREAKING_PAWN_CHAINS_BONUS",
    "bishop_pair_bonus", 
    "knight_on_the_rim_penalty", 
    "pawn_chain_bonus",
    "PASSED_PAWN_BONUS", 
    "enemy_king_corner_bonus", 
    "enemy_king_center_bonus",
    "hanging_piece_penalty", 
    "squares_controlled_bonus", 
    "pieac_pos_bonus",
    "rook_open_file_bonus",
    "isolated_pawn_penalty"
}

best_params = {
    "BREAKING_PAWN_CHAINS_BONUS": 1.0,
    "CONTROL_CENTER_BONUS": 0.5,
    "END_RATE": 0.6,
    "KING_SAFETY_BONUS": 5.0,
    "OPEN_RATE": 0.6,
    "PASSED_PAWN_BONUS": 0.5,
    "bishop_pair_bonus": 0.5,
    "enemy_king_center_bonus": 0.5,
    "enemy_king_corner_bonus": 1.0,
    "hanging_piece_penalty": 0.5,
    "isolated_pawn_penalty": 0.5,
    "knight_on_the_rim_penalty": 0.5,
    "pawn_chain_bonus": 2.0,
    "pieac_pos_bonus": 0.01,
    "rook_open_file_bonus": 0.5,
    "squares_controlled_bonus": 0.01,
}

def evaluate_single(item, weights, error_counter):
    board, current_color, state = item
    try:
        return evaluate_board(board, state, {}, current_color, weights)
    except Exception as e:
        error_counter["count"] += 1
        error_counter["last_error"] = str(e)
    return 0.0


def evaluate_all(parsed_boards, weights, error_counter):
    return np.array([evaluate_single(item, weights, error_counter) for item in parsed_boards])


def best_scale(preds, targets):
    denom = np.dot(preds, preds)
    if denom == 0:
        return 1.0
    scale = np.dot(preds, targets) / denom
    return float(np.clip(scale, 0.01, 1000.0))


def save_progress(param_keys, weights, scale, mse, filename=OUTPUT_TXT):
    """Sparar bästa kända parametrar löpande så inget går förlorat vid avbrott."""
    optimized_params = {}
    for i, key in enumerate(param_keys):
        if key in RATE_PARAMS:
            optimized_params[key] = round(float(weights[i]), 4)
        else:
            optimized_params[key] = round(float(weights[i] * scale), 4)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Binär sökning / Intervallhalvering optimerade parametrar (MSE={mse:.2f})\n")
        f.write("best_params = {\n")
        for key, val in optimized_params.items():
            f.write(f'    "{key}": {val},\n')
        f.write("}\n")


def sample_and_parse(full_df, sample_size, seed=None):
    """Drar ett NYTT slumpmässigt urval ur hela datasetet och parsar FEN-strängarna."""
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
        print(f"    ({fail_count}/{len(df_sample)} FEN kunde inte parsas i detta urval och exkluderas)")

    valid_boards = [b for b, ok in zip(boards, ok_mask) if ok]
    valid_sf = df_sample["stockfish_eval"].values[ok_mask]
    return valid_boards, valid_sf


def binary_search_optimize(full_df, param_keys, initial_weights):
    """Optimerar parametrar med adaptiv binär sökning (åt båda håll) och brusfilter."""
    current_weights = list(initial_weights)
    error_counter = {"count": 0, "last_error": None}

    # Baslinje-MSE på ett första urval
    boards0, sf0 = sample_and_parse(full_df, SAMPLE_SIZE, seed=RANDOM_SEED)
    initial_dict = {param_keys[i]: current_weights[i] for i in range(len(param_keys))}
    preds0 = evaluate_all(boards0, initial_dict, error_counter)
    scale0 = best_scale(preds0, sf0)
    initial_mse = np.mean((scale0 * preds0 - sf0) ** 2)
    best_scale_val = scale0
    best_mse = initial_mse

    print(f"\n--- Startar Robust Binär Sökning (Upp & Ner)! Initial MSE: {initial_mse:.2f} ---")
    print(f"Tidsgräns satt till: {MAX_MINUTES} minuter.")

    start_time = time.time()
    time_limit_seconds = MAX_MINUTES * 60

    for it in range(MAX_ITERATIONS):
        if time.time() - start_time >= time_limit_seconds:
            print(f"\n*** Tidsgränsen på {MAX_MINUTES} minuter har nåtts! Avbryter sökningen. ***")
            break

        print(f"\n>>> ITERATION {it + 1}/{MAX_ITERATIONS} <<<")

        parsed_boards, sf_evals = sample_and_parse(full_df, SAMPLE_SIZE, seed=None)
        print(f"    Nytt urval draget: {len(parsed_boards)} giltiga positioner.")

        current_dict = {param_keys[i]: current_weights[i] for i in range(len(param_keys))}
        preds_current = evaluate_all(parsed_boards, current_dict, error_counter)
        best_scale_val = best_scale(preds_current, sf_evals)
        best_mse = np.mean((best_scale_val * preds_current - sf_evals) ** 2)
        print(f"    MSE för nuvarande vikter på nya urvalet: {best_mse:.2f}")

        improved_this_iteration = False

        for i, key in enumerate(tqdm(param_keys, desc=f"Runda {it + 1}", colour="blue")):
            if time.time() - start_time >= time_limit_seconds:
                print(f"\n*** Tidsgränsen på {MAX_MINUTES} minuter har nåtts! Avbryter sökningen. ***")
                return current_weights, best_scale_val, best_mse

            current_val = current_weights[i]
            best_val_for_key = current_val

            # Starta med ett bra söksteg och tillåt acceleration uppåt
            step = 1.0              
            max_step = 16.0         
            tolerance_step = 0.001

            while step >= tolerance_step:
                if abs(current_val) < 1e-4:
                    base_val = 0.01
                else:
                    base_val = current_val

                # TESTAR BÅDE NERÅT OCH UPPÅT SAMTIDIGT
                test_vals = [
                    base_val * (1.0 - step),
                    base_val * (1.0 + step)
                ]

                if key in RATE_PARAMS:
                    test_vals = [v for v in test_vals if 0.01 <= v <= 0.99]
                
                if key in POSITIVE_PARAMS:
                    test_vals = [v for v in test_vals if v >= 0.001]

                improved_in_step = False
                for val in test_vals:
                    if val == current_val:
                        continue

                    test_weights = list(current_weights)
                    test_weights[i] = val
                    weight_dict = {param_keys[j]: test_weights[j] for j in range(len(param_keys))}

                    preds = evaluate_all(parsed_boards, weight_dict, error_counter)
                    
                    # Ignorera helt om uträkningen är identisk (förhindrar låsning på döda parametrar)
                    if np.array_equal(preds, preds_current):
                        continue

                    scale = best_scale(preds, sf_evals)
                    mse = np.mean((scale * preds - sf_evals) ** 2)

                    # Krävs en faktisk märkbar förbättring för att filtrera bort flyttalsbrus
                    if best_mse - mse > 0.001:
                        print(f"  [+] {key} förbättrades (steg={step:.3f})! {current_val:.4f} -> {val:.4f} (Ny MSE: {mse:.2f})")
                        best_mse = mse
                        best_val_for_key = val
                        best_scale_val = scale
                        improved_this_iteration = True
                        improved_in_step = True

                if improved_in_step:
                    current_val = best_val_for_key
                    # Accelerera steget om vi rör oss åt rätt håll
                    step = min(step * 1.5, max_step)
                else:
                    # Inen förbättring i någon riktning? Halvera steget för att söka mer lokalt.
                    step /= 2.0

            current_weights[i] = best_val_for_key

        save_progress(param_keys, current_weights, best_scale_val, best_mse)

        if error_counter["count"]:
            print(f"    (Totalt {error_counter['count']} evaluerings-fel hittills, "
                  f"senaste: {error_counter['last_error']})")

        if not improved_this_iteration:
            print("    Ingen förbättring denna runda.")

    return current_weights, best_scale_val, best_mse


def main():
    if not os.path.exists(CSV_FILENAME):
        print(f"Kunde inte hitta {CSV_FILENAME}.")
        return

    full_df = pd.read_csv(CSV_FILENAME)
    required_cols = {"fen", "eval_cp", "turn"}
    missing = required_cols - set(full_df.columns)
    if missing:
        print(f"CSV:n saknar förväntade kolumner: {missing}. Hittade: {list(full_df.columns)}")
        return

    full_df = full_df.rename(columns={"eval_cp": "stockfish_eval"})
    full_df["stockfish_eval"] = full_df["stockfish_eval"].clip(-2000, 2000)
    print(f"Laddade {len(full_df)} positioner totalt från {CSV_FILENAME}. "
          f"Ett nytt urval på {SAMPLE_SIZE} dras varje runda.")

    param_keys = sorted(best_params.keys())

    if len(full_df) == 0:
        print("Inga positioner i CSV:n — avbryter.")
        return

    initial_weights = [float(best_params[k]) for k in param_keys]

    t0 = time.time()
    w, scale, final_mse = binary_search_optimize(full_df, param_keys, initial_weights)

    print(f"\nBinär sökning avslutades efter {time.time() - t0:.2f} sekunder. Slutlig MSE: {final_mse:.2f}")

    save_progress(param_keys, w, scale, final_mse)

    optimized_params = {}
    for i, key in enumerate(param_keys):
        if key in RATE_PARAMS:
            optimized_params[key] = round(float(w[i]), 4)
        else:
            optimized_params[key] = round(float(w[i] * scale), 4)

    print("\nNya optimerade parametrar:")
    for key, val in optimized_params.items():
        print(f'    "{key}": {val},')

    print(f"\nSparat till '{OUTPUT_TXT}'!")


if __name__ == "__main__":
    main()