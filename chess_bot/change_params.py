"""
Optimerar viktparametrarna i din schack-evalueringsfunktion mot Stockfish-data.

Funktionen är icke-linjär (för många parametrar + villkorslogik i evalueringen
för att en linjär modell ska stämma), så vi kör en enda, bra väg:

1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) för sökningen.
   Detta är i praktiken standardvalet för gradientfri optimering av
   svartlåde-funktioner med tiotals parametrar - betydligt bättre skalning
   och konvergens än Powell/Nelder-Mead, och den utvärderar en hel
   population varje generation vilket parallelliseras naturligt över dina
   CPU-kärnor.
2. Skalfaktorn (hur mycket vår output ska multipliceras med för att matcha
   Stockfish centipawns) räknas ut analytiskt för varje kandidat-vikt istället
   för att vara en egen parameter att söka efter - exakt och gratis, en
   dimension mindre att optimera.
3. Train/val-split så du ser om du overfittar.
4. Ridge-regularisering (valfri) som drar vikterna mot startvärdena.
5. Dynamiska bounds per parameter (0.1x - 10x av nuvarande värde) istället
   för en hårdkodad (0.01, 20.0) som kan vara helt fel skala för dina
   parametrar.
6. Sparar .txt (med skalfaktorn inbakad, så din engine kan läsa den direkt
   utan att behöva veta något om scale) och .json med diagnostik.

Kräver: pip install cma

Användning:
    python change_params.py
    python change_params.py --sample-size 2000 --reg-lambda 0.01
"""

import os
import json
import time
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import cma
except ImportError:
    raise SystemExit(
        "Paketet 'cma' saknas. Installera med: pip install cma"
    )

try:
    from main import parse_fen, evaluate_board, best_params
except ImportError:
    print("Varning: Kunde inte importera från main.py. Kontrollera att filerna ligger i samma mapp.")


# ---------------------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------------------

def load_data(csv_filename, sample_size, seed=42):
    df = pd.read_csv(csv_filename)
    if df.empty:
        raise ValueError("CSV-filen är tom.")

    fixed_columns = {"game", "move", "turn", "our_eval", "stockfish_eval", "fen"}
    param_keys = [c for c in df.columns if c not in fixed_columns]
    if not param_keys:
        raise ValueError("Inga parametrar hittades i CSV-filen.")

    df = df.copy()
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
    """Analytisk minsta-kvadrat-skala: minimerar ||scale*preds - targets||^2."""
    denom = np.dot(preds, preds)
    return (np.dot(preds, targets) / denom) if denom != 0 else 1.0


# ---------------------------------------------------------------------------
# CMA-ES-optimering (+ valfri multiprocessing för parallell utvärdering)
# ---------------------------------------------------------------------------

# Worker-globaler: sätts en gång per process av _pool_init, återanvänds för
# varje kandidat som skickas in. Detta undviker att skicka om boards/sf_evals
# över IPC för varje utvärdering.
_WORKER_BOARDS = None
_WORKER_SF = None
_WORKER_REG_LAMBDA = None
_WORKER_X0 = None


def _pool_init(parsed_boards, sf_evals, reg_lambda, x0):
    global _WORKER_BOARDS, _WORKER_SF, _WORKER_REG_LAMBDA, _WORKER_X0
    _WORKER_BOARDS = parsed_boards
    _WORKER_SF = sf_evals
    _WORKER_REG_LAMBDA = reg_lambda
    _WORKER_X0 = x0


def _pool_eval_candidate(args):
    """
    Kör i en worker-process: utvärderar EN hel kandidat (alla positioner)
    lokalt, utan ytterligare IPC. Detta är den viktiga skillnaden mot
    tidigare version, där varje kandidat delades upp i bitar och skickades
    ut till alla workers - det gav en pool.map-runda PER kandidat istället
    för en pool.map-runda PER GENERATION (dvs. popsize gånger fler
    IPC-rundor än nödvändigt).
    """
    weight_dict, x_real = args
    preds = evaluate_all(_WORKER_BOARDS, weight_dict)
    scale = best_scale(preds, _WORKER_SF)
    mse = np.mean((scale * preds - _WORKER_SF) ** 2)
    if _WORKER_REG_LAMBDA > 0:
        mse += _WORKER_REG_LAMBDA * np.mean((x_real - _WORKER_X0) ** 2)
    return mse


def solve_nonlinear(parsed_boards, param_keys, sf_evals, bounds, initial_weights,
                     max_evals=4000, n_jobs=1, reg_lambda=0.0, seed=42):
    """
    Optimerar med CMA-ES. Sökrymden är centrerad kring initial_weights (skalad
    till ungefär enhetslängd internt, vilket CMA-ES trivs bäst med). Hela
    populationen för en generation fördelas över workers i EN pool.map-runda -
    varje worker äger en komplett kandidat och utvärderar alla positioner
    själv, vilket minimerar antalet IPC-rundor drastiskt jämfört med att dela
    upp positionerna per kandidat.
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.array(initial_weights, dtype=float)

    # CMA-ES vill ha en enda global sigma - vi normaliserar parametrarna med
    # deras egna bound-bredd så att ett steg betyder "ungefär lika mycket"
    # för varje parameter oavsett dess råa skala (t.ex. bonde=100 vs mobilitet=0.1).
    scale_per_param = np.maximum(hi - lo, 1e-6)

    def to_real(x_norm):
        return lo + (x_norm * scale_per_param)

    def to_norm(x_real):
        return (x_real - lo) / scale_per_param

    x0_norm = to_norm(x0)
    norm_bounds = [np.zeros(len(param_keys)), np.ones(len(param_keys))]

    pool = None
    if n_jobs > 1:
        pool = mp.Pool(n_jobs, initializer=_pool_init,
                        initargs=(parsed_boards, sf_evals, reg_lambda, x0))

    pbar = tqdm(total=max_evals, desc="Optimerar (CMA-ES)", colour="green")
    count = [0]

    def solutions_to_tasks(solutions):
        tasks = []
        for x_norm in solutions:
            x_real = to_real(np.clip(x_norm, 0.0, 1.0))
            weight_dict = {param_keys[i]: float(x_real[i]) for i in range(len(param_keys))}
            tasks.append((weight_dict, x_real))
        return tasks

    def objective_serial(weight_dict, x_real):
        preds = evaluate_all(parsed_boards, weight_dict)
        scale = best_scale(preds, sf_evals)
        mse = np.mean((scale * preds - sf_evals) ** 2)
        if reg_lambda > 0:
            mse += reg_lambda * np.mean((x_real - x0) ** 2)
        return mse

    es = cma.CMAEvolutionStrategy(
        x0_norm, 0.25,  # startsigma på den normaliserade skalan (0-1)
        {"bounds": norm_bounds, "seed": seed, "verbose": -9},
    )

    while not es.stop() and count[0] < max_evals:
        solutions = es.ask()
        tasks = solutions_to_tasks(solutions)

        if pool is not None:
            fitnesses = pool.map(_pool_eval_candidate, tasks)
        else:
            fitnesses = [objective_serial(wd, xr) for wd, xr in tasks]

        count[0] += len(solutions)
        pbar.update(len(solutions))
        es.tell(solutions, fitnesses)

    pbar.close()
    if pool is not None:
        pool.close()
        pool.join()

    x_best_norm = es.result.xbest
    x_best = to_real(x_best_norm)
    weight_dict = {param_keys[i]: float(x_best[i]) for i in range(len(param_keys))}
    preds = evaluate_all(parsed_boards, weight_dict)
    scale = best_scale(preds, sf_evals)
    return x_best, scale


# ---------------------------------------------------------------------------
# Huvudfunktion
# ---------------------------------------------------------------------------

def optimize_parameters(csv_filename="evaluation_comparison.csv", sample_size=1000,
                         max_evals=4000, n_jobs=None, reg_lambda=0.0, val_split=0.2):
    if not os.path.exists(csv_filename):
        print(f"Hittade ingen fil med namnet {csv_filename}. Kör några matcher först!")
        return

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    df_sample, param_keys = load_data(csv_filename, sample_size)
    print(f"Laddade {len(df_sample)} positioner, {len(param_keys)} parametrar: {param_keys}")

    n_val = int(len(df_sample) * val_split)
    df_val = df_sample.iloc[:n_val]
    df_train = df_sample.iloc[n_val:]

    print("Parsar schackbräden...")
    t0 = time.time()
    train_boards = parse_positions(df_train["fen"].values)
    val_boards = parse_positions(df_val["fen"].values) if n_val > 0 else []
    print(f"  klart på {time.time() - t0:.1f}s")

    try:
        initial_weights = [float(best_params[k]) for k in param_keys]
        print("Laddade startvikter direkt från main.py (best_params).")
    except (NameError, KeyError):
        print("Varning: Kunde inte hämta best_params från main.py. Fallback till CSV-rad.")
        initial_weights = [float(df_sample.iloc[-1][k]) for k in param_keys]

    # Dynamiska bounds: 0.1x - 10x av nuvarande värde per parameter
    bounds = []
    for w in initial_weights:
        if w > 0:
            bounds.append((w * 0.1, w * 10.0))
        elif w < 0:
            bounds.append((w * 10.0, w * 0.1))
        else:
            bounds.append((-10.0, 10.0))

    train_sf = df_train["stockfish_eval"].values
    val_sf = df_val["stockfish_eval"].values if n_val > 0 else np.array([])

    initial_dict = {param_keys[i]: initial_weights[i] for i in range(len(param_keys))}
    before_preds = evaluate_all(train_boards, initial_dict)
    before_scale = best_scale(before_preds, train_sf)
    before_mse = np.mean((before_scale * before_preds - train_sf) ** 2)

    print(f"\nKör CMA-ES-optimering ({n_jobs} parallella processer, upp till {max_evals} utvärderingar)...")
    t0 = time.time()
    w, scale = solve_nonlinear(train_boards, param_keys, train_sf, bounds,
                                initial_weights, max_evals=max_evals, n_jobs=n_jobs,
                                reg_lambda=reg_lambda)
    elapsed = time.time() - t0
    print(f"Optimering klar på {elapsed:.2f} sekunder.")

    raw_optimized_params = {param_keys[i]: float(w[i]) for i in range(len(param_keys))}
    optimized_params = {param_keys[i]: round(float(w[i] * scale), 4) for i in range(len(param_keys))}

    after_preds = evaluate_all(train_boards, raw_optimized_params)
    after_mse = np.mean((scale * after_preds - train_sf) ** 2)

    print("\n---------------------------------------------------")
    print(f"Tränings-MSE innan: {before_mse:.1f}")
    print(f"Tränings-MSE efter: {after_mse:.1f}", end="")
    if before_mse > 0:
        print(f"  ({(1 - after_mse / before_mse) * 100:.1f}% bättre)")
    else:
        print()

    if n_val > 0:
        val_preds = evaluate_all(val_boards, raw_optimized_params)
        val_mse = np.mean((scale * val_preds - val_sf) ** 2)
        corr = np.corrcoef(scale * val_preds, val_sf)[0, 1] if np.std(val_preds) > 0 else float("nan")
        print(f"Validerings-MSE (osedda positioner): {val_mse:.1f}")
        print(f"Korrelation med Stockfish (validering): {corr:.3f}")
        if val_mse > after_mse * 1.5:
            print("OBS: Validerings-MSE är klart sämre än tränings-MSE -> risk för overfitting.")
            print("     Prova --reg-lambda 0.01 (eller högre) för att dra vikterna mot startvärdena.")

    print("\nNya parametrar (med inbakad skalfaktor):")
    for key, val in optimized_params.items():
        print(f'    "{key}": {val},')

    with open("optimized_params.txt", "w", encoding="utf-8") as f:
        f.write("# Optimerade parametrar baserade på Stockfish-analys (CMA-ES, scale inbakad)\n")
        f.write("best_params = {\n")
        for key, val in optimized_params.items():
            f.write(f'    "{key}": {val},\n')
        f.write("}\n")

    with open("optimized_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "params": optimized_params,
            "scale": float(scale),
            "train_mse_before": float(before_mse),
            "train_mse_after": float(after_mse),
            "method": "cma-es",
        }, f, indent=2, ensure_ascii=False)

    print("\nSparat till 'optimized_params.txt' och 'optimized_params.json'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="evaluation_comparison.csv")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=4000,
                         help="Max antal funktionsutvärderingar för CMA-ES.")
    parser.add_argument("--n-jobs", type=int, default=None,
                         help="Antal parallella processer (default: alla kärnor - 1).")
    parser.add_argument("--reg-lambda", type=float, default=0.0,
                         help="Ridge-regularisering mot startvikterna, t.ex. 0.01. 0 = av.")
    args = parser.parse_args()

    optimize_parameters(
        csv_filename=args.csv,
        sample_size=args.sample_size,
        max_evals=args.max_evals,
        n_jobs=args.n_jobs,
        reg_lambda=args.reg_lambda,
    )