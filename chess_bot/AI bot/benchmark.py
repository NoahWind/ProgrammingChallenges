import time
import chess
import torch

# Importera din nuvarande sök-kod (vi antar att den ligger i search.py)
import search

# Importera din modell
try:
    from nn_model import ChessEvaluatorNN
except ImportError:
    print("Kunde inte importera ChessEvaluatorNN. Använder en dummy-modell för benchmark.")
    import torch.nn as nn
    class DummyModel(nn.Module):
        def forward(self, x):
            # Returnerar slumpmässiga värden mellan -1 och 1 för test
            return torch.rand(x.size(0), 1) * 2 - 1
    ChessEvaluatorNN = DummyModel

def run_old_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kör benchmark för gamla koden på enhet: {device.upper()}\n")

    # Ladda modellen
    model = ChessEvaluatorNN()
    model.to(device)
    model.eval()

    # Tre klassiska benchmark-ställningar
    test_positions = {
        "Startposition": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Komplicerat mittspel (KiwiPete)": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "Slutspel": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    }

    # Djup att testa
    depths = [2, 3, 4]

    for name, fen in test_positions.items():
        print(f"=== Testar: {name} ===")
        board = chess.Board(fen)

        for depth in depths:
            # Rensa cachen inför varje sökning för att få en exakt och rättvis tidsmätning
            search.clear_search_caches()
            
            start_time = time.perf_counter()
            best_move, score = search.find_best_move(board, model, depth=depth, device=device)
            elapsed_time = time.perf_counter() - start_time

            print(f"  Djup {depth}: {elapsed_time:.4f} sekunder | Bästa drag: {best_move} (Poäng: {score:.2f})")
        print("-" * 40)

if __name__ == "__main__":
    run_old_benchmark()