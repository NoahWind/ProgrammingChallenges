"""
check_buffer_material_balance.py
---------------------------------
Skannar din sparade replay_buffer.pt och rapporterar hur väl representerade
positioner med STOR materialobalans (t.ex. hängande pjäser) är i träningsdatan.

Om t.ex. <2% av positionerna har |materialbalans| >= 3 (ungefär en lätt pjäs),
är det en stark ledtråd till varför nätverket inte lärt sig att straffa
hängande pjäser ordentligt -- det har helt enkelt sett väldigt få sådana
exempel jämfört med "normala" balanserade mittspelspositioner.

Kör:
    python check_buffer_material_balance.py --buffer-path replay_buffer.pt
"""
import argparse
import torch

from nn_model import INPUT_SIZE

PIECE_VALUES = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0] * 2)  # vit x6, svart x6


def material_balance(X: torch.Tensor) -> torch.Tensor:
    """Räknar exakt materialbalans (vitt - svart, i bönder) direkt ur
    board_to_tensor-kodningen, oberoende av nätverket."""
    planes = X[:, :768].view(-1, 12, 64)
    counts = planes.sum(dim=2)  # (N, 12)
    white = (counts[:, :6] * PIECE_VALUES[:6]).sum(dim=1)
    black = (counts[:, 6:] * PIECE_VALUES[6:]).sum(dim=1)
    return white - black


def bucket_report(values: torch.Tensor, edges, label: str):
    n = values.shape[0]
    print(f"\n{label} (n={n}):")
    for lo, hi in edges:
        if hi is None:
            mask = values.abs() >= lo
            name = f"|x| >= {lo}"
        else:
            mask = (values.abs() >= lo) & (values.abs() < hi)
            name = f"{lo} <= |x| < {hi}"
        cnt = int(mask.sum().item())
        pct = 100 * cnt / n if n else 0
        bar = "#" * int(pct / 2)
        print(f"  {name:16s}: {cnt:7d}  ({pct:5.1f}%)  {bar}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer-path", type=str, default="replay_buffer.pt")
    args = p.parse_args()

    data = torch.load(args.buffer_path)
    X, y = data["X"], data["y"]
    print(f"Laddade buffer: {X.shape[0]} positioner.")

    if X.shape[1] != INPUT_SIZE:
        print(f"VARNING: X har {X.shape[1]} kolumner, förväntade {INPUT_SIZE}. "
              f"Kodningen kan ha ändrats sedan bufferten skapades.")

    mat = material_balance(X)

    edges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 9), (9, None)]
    bucket_report(mat, edges, "Materialbalans (räknad direkt ur brädet, i bönder)")

    edges_y = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 9), (9, None)]
    bucket_report(y, edges_y, "Stockfish-facit |y| (pawns, klippt vid 10 i träning)")

    big_mat_frac = (mat.abs() >= 3).float().mean().item()
    print(f"\n>>> Andel positioner med |materialbalans| >= 3 (ca en lätt pjäs eller mer): "
          f"{100 * big_mat_frac:.1f}%")
    if big_mat_frac < 0.05:
        print(">>> Det är LÅGT. Nätverket har sett relativt få tydliga "
              "materialobalanser -- rimlig delförklaring till att det inte "
              "straffar hängande pjäser konsekvent.")
    else:
        print(">>> Det är en rimlig andel -- underrepresentation är då mindre "
              "trolig som huvudorsak; kolla snarare specifika taktiska mönster.")

    # Extra: bland de mest extrema materialobalanserna, hur väl matchar
    # nätverkets facit (y) den faktiska materialbalansen? Stora avvikelser
    # kan avslöja brusiga/svaga Stockfish-facit vid grunt sökdjup.
    extreme_mask = mat.abs() >= 5
    if extreme_mask.sum() > 0:
        diff = (y[extreme_mask] - mat[extreme_mask]).abs()
        print(f"\nBland {int(extreme_mask.sum())} positioner med |materialbalans| >= 5:")
        print(f"  Genomsnittlig |facit - materialbalans| = {diff.mean().item():.2f} pawns")
        print(f"  (Stora värden kan betyda att facit-djupet var för grunt för att "
              f"fånga kompensation/hot korrekt i dessa ställningar.)")


if __name__ == "__main__":
    main()
