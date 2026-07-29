"""
Regressionsbaserad optimering av schack-parametrar.
========================================================

VARFÖR DEN GAMLA METODEN (binär sökning / CMA-ES / Nelder-Mead) INTE
KOMMER SÅ NÄRA:

Alla dessa metoder är giriga, lokala sökmetoder som testar en (eller ett
fåtal) parametrar i taget och hoppas hitta vägen till ett bra minimum.
Med 17 parametrar som interagerar via samma fas-vikter (a, b, c) fastnar
de lätt i lokala minimum, och 20 min / 200000 iterationer räcker sällan
för att verkligen konvergera med bra precision.

INSIKTEN SOM GÖR DET HÄR MYCKET BÄTTRE:

Om man läser evaluate_board() i rating.py ser man att VARJE heuristik-term
har formen:

    score += <bool/kontinuerlig feature> * ENGINE_PARAMS["some_key"] * (a + b + c)

dvs en enda parameter multiplicerad med en boolesk/kontinuerlig feature,
ALDRIG två olika tunable-parametrar multiplicerade med varandra. Det gör
att hela evalueringen (minus material_score och de icke-tunable endgame-
bonusarna) är EXAKT LINJÄR i de 15 heuristik-vikterna, för FASTA värden
på OPEN_RATE och END_RATE (som styr a/b/c olinjärt).

STRATEGI:

1. OPEN_RATE och END_RATE söks igenom med ett grovt->förfinat grid
   (de går in olinjärt, så äkta grid-sök är rätt verktyg här).
2. För VARJE (OPEN_RATE, END_RATE)-kombination löses de 15 övriga
   vikterna EXAKT med minsta-kvadrat-regression (bundna, icke-negativa)
   istället för att gissa oss fram. Vi hittar varje parameters bidrag
   per position genom ett "impulssvar": sätt parametern till 1, alla
   andra tunable till 0, kör evaluate_board, se hur mycket poängen
   ändras. Det ÄR den positionens koefficient för den parametern -
   fungerar oavsett hur komplex/asymmetrisk formeln råkar vara internt,
   så länge den är linjär i just den parametern (vilket den är här).
3. Den (OPEN_RATE, END_RATE, w)-kombo som ger lägst MAE vinner.

Detta ger GARANTERAT globalt optimum för given (OPEN_RATE, END_RATE),
på sekunder istället för minuter - och vi behöver aldrig gissa steglängder.

VIKTIG ÄRLIG BRASK-LAPP:
Om MAE fortfarande inte blir så låg som du vill efter detta, är
flaskhalsen inte längre SÖKMETODEN (den är nu exakt/optimal) utan
MODELLENS UTTRYCKSKRAFT: en linjär kombination av 15 booleska/kontinuerliga
features med gemensam fas-blandning (a+b+c) kan helt enkelt inte fånga
allt Stockfish "ser" (den gör full sökning + NN-eval). Då är nästa steg
att lägga till FLER/bättre features i different_evals.py, inte att byta
sökmetod igen.
"""

import os
import time
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from tqdm import tqdm

try:
    from main import parse_fen
    from rating import evaluate_board
except ImportError:
    print("Varning: Kunde inte importera parse_fen/evaluate_board. Körs i fel mapp?")

# =====================================================================
# 1. INSTÄLLNINGAR
# =====================================================================
CSV_FILENAME = os.path.join("stockfish_self_play_results", "positions_ratings.csv")
SAMPLE_SIZE = None          # None = hela filen
OUTPUT_TXT = "optimized_regression.txt"

# Alla parametrar UTOM OPEN_RATE/END_RATE - de löses linjärt via regression.
TUNABLE_KEYS = sorted([
    "BREAKING_PAWN_CHAINS_BONUS",
    "CONTROL_CENTER_BONUS",
    "KING_SAFETY_BONUS",
    "PASSED_PAWN_BONUS",
    "bishop_pair_bonus",
    "enemy_king_center_bonus",
    "enemy_king_corner_bonus",
    "hanging_piece_penalty",
    "isolated_pawn_penalty",
    "knight_on_the_rim_penalty",
    "pawn_chain_bonus",
    "pieac_pos_bonus",
    "rook_open_file_bonus",
    "squares_controlled_bonus",
    "rook_on_seventh_rank_bonus",
])

# Undre gränser (samma andemening som i din gamla MIN_BOUNDS). Övre gräns = oändlighet.
MIN_BOUNDS = {
    "KING_SAFETY_BONUS": 0.0,
    "CONTROL_CENTER_BONUS": 0.0,
    "BREAKING_PAWN_CHAINS_BONUS": 0.0,
    "bishop_pair_bonus": 0.0,
    "knight_on_the_rim_penalty": 0.0,
    "pawn_chain_bonus": 0.0,
    "PASSED_PAWN_BONUS": 0.0,
    "enemy_king_corner_bonus": 0.0,
    "enemy_king_center_bonus": 0.0,
    "hanging_piece_penalty": 0.0,
    "squares_controlled_bonus": 0.0,
    "pieac_pos_bonus": 0.0,
    "rook_open_file_bonus": 0.0,
    "isolated_pawn_penalty": 0.0,
    "rook_on_seventh_rank_bonus": 0.0,
}

# =====================================================================
# 2. DATAHANTERING (samma som förut)
# =====================================================================

def sample_and_parse(full_df, sample_size, seed=None):
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
# 3. IMPULSSVAR -> DESIGNMATRIS
# =====================================================================

def _eval_worker(args):
    """Måste ligga top-level för multiprocessing."""
    board_state, weights = args
    board, current_color, state = board_state
    try:
        return evaluate_board(board, state, {}, current_color, weights)
    except Exception:
        return 0.0

def build_design_matrix(parsed_boards, open_rate, end_rate, pool):
    """
    Bygger X (n_positioner x n_parametrar) sådan att:
        evaluate_board(pos, w) ~= baseline(pos) + X[pos] @ w
    Görs genom att mäta MARGINALEFFEKTEN av varje parameter separat
    (sätt den till 1, resten till 0), vilket är EXAKT eftersom
    evaluate_board är linjär i varje enskild tunable-parameter.
    """
    n = len(parsed_boards)
    k = len(TUNABLE_KEYS)

    base_weights = {key: 0.0 for key in TUNABLE_KEYS}
    base_weights["OPEN_RATE"] = open_rate
    base_weights["END_RATE"] = end_rate

    all_weight_dicts = [base_weights] + [
        {**base_weights, key: 1.0} for key in TUNABLE_KEYS
    ]

    # En stor batch-körning: (1 + k) * n evalueringar totalt, parallellt.
    tasks = [(bs, wd) for wd in all_weight_dicts for bs in parsed_boards]
    flat_results = np.array(pool.map(_eval_worker, tasks), dtype=np.float64)
    results = flat_results.reshape(len(all_weight_dicts), n)

    baseline = results[0]
    X = (results[1:] - baseline[None, :]).T  # shape (n, k)
    return X, baseline

def solve_for_rates(parsed_boards, sf_evals, open_rate, end_rate, pool):
    X, baseline = build_design_matrix(parsed_boards, open_rate, end_rate, pool)
    y = sf_evals - baseline

    lb = np.array([MIN_BOUNDS.get(k, 0.0) for k in TUNABLE_KEYS])
    ub = np.full(len(TUNABLE_KEYS), np.inf)

    result = lsq_linear(X, y, bounds=(lb, ub), method="trf", lsq_solver="lsmr", max_iter=200)
    w = result.x
    preds = baseline + X @ w
    mae = float(np.mean(np.abs(preds - sf_evals)))
    return w, mae

# =====================================================================
# 4. GRID-SÖKNING ÖVER OPEN_RATE / END_RATE (grov -> förfinad)
# =====================================================================

def scan_grid(parsed_boards, sf_evals, pool, open_values, end_values, desc):
    best = None
    combos = list(itertools.product(open_values, end_values))
    for open_rate, end_rate in tqdm(combos, desc=desc, colour="green"):
        w, mae = solve_for_rates(parsed_boards, sf_evals, open_rate, end_rate, pool)
        if best is None or mae < best[0]:
            best = (mae, float(open_rate), float(end_rate), w)
            tqdm.write(f"    [+] Ny bästa: OPEN_RATE={open_rate:.4f} END_RATE={end_rate:.4f} MAE={mae:.4f}")
    return best

def save_progress(open_rate, end_rate, w, mae, filename=OUTPUT_TXT):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Regression-optimerade parametrar (MAE={mae:.4f})\n")
        f.write("best_params = {\n")
        f.write(f'    "OPEN_RATE": {round(float(open_rate), 4)},\n')
        f.write(f'    "END_RATE": {round(float(end_rate), 4)},\n')
        for key, val in zip(TUNABLE_KEYS, w):
            f.write(f'    "{key}": {round(float(val), 4)},\n')
        f.write("}\n")

# =====================================================================
# 5. MAIN
# =====================================================================

def main():
    if not os.path.exists(CSV_FILENAME):
        print(f"Kunde inte hitta {CSV_FILENAME}.")
        return

    full_df = pd.read_csv(CSV_FILENAME)
    full_df = full_df.rename(columns={"eval_cp": "stockfish_eval"})
    full_df["stockfish_eval"] = full_df["stockfish_eval"].clip(-1500, 1500) / 100.0

    print("Laddar och fryser statiskt träningsurval...")
    parsed_boards, sf_evals = sample_and_parse(full_df, SAMPLE_SIZE, seed=42)
    print(f"Klart: {len(parsed_boards)} positioner.")

    print("antal cores:", mp.cpu_count())
    num_cores = 4
    print(f"Använder {num_cores} CPU-kärnor.")

    t0 = time.time()
    with mp.Pool(processes=num_cores) as pool:
        # --- Steg 1: grovt grid över hela intervallet ---
        coarse = np.array([0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
        mae, open_rate, end_rate, w = scan_grid(
            parsed_boards, sf_evals, pool, coarse, coarse, desc="Grovt grid"
        )
        print(f"\nGrovt grid klart: OPEN_RATE={open_rate:.4f} END_RATE={end_rate:.4f} MAE={mae:.4f}")
        save_progress(open_rate, end_rate, w, mae)

        # --- Steg 2: förfinat grid runt bästa punkten ---
        span = 0.08
        steps = 9
        open_values = np.clip(np.linspace(open_rate - span, open_rate + span, steps), 0.001, 0.99)
        end_values = np.clip(np.linspace(end_rate - span, end_rate + span, steps), 0.001, 0.99)
        mae2, open_rate2, end_rate2, w2 = scan_grid(
            parsed_boards, sf_evals, pool, open_values, end_values, desc="Förfinat grid"
        )
        if mae2 < mae:
            mae, open_rate, end_rate, w = mae2, open_rate2, end_rate2, w2

        print(f"\nFörfinat grid klart: OPEN_RATE={open_rate:.4f} END_RATE={end_rate:.4f} MAE={mae:.4f}")
        save_progress(open_rate, end_rate, w, mae)

    print(f"\nOptimering klar på {time.time() - t0:.1f} sekunder. Slutlig MAE: {mae:.4f}")
    print(f"Resultat sparat i {OUTPUT_TXT}")

if __name__ == "__main__":
    main()