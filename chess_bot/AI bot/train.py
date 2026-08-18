"""
train.py
--------
Tränar ChessEvaluatorNN direkt från den lokala Parquet-databasen.
Strömmar datan i batchar för att spara RAM-minne.
"""

from __future__ import annotations
import argparse
import os
import time
import csv
from xml.parsers.expat import model

import chess
import pyarrow.dataset as ds
import torch
import torch.nn as nn

from nn_model import ChessEvaluatorNN, board_to_tensor, save_model, load_model

normal_databas = "min_schack_databas/lichess_data_converted.parquet"  # Standarddatabas med riktiga ställningar
test = "min_schack_databas/kq_vs_k_positions.parquet"  # Testdatabas med färre ställningar
test_2 = "min_schack_databas/kq_mates_1_2.parquet"  # Testdatabas med färre ställningar

def parse_args():
    p = argparse.ArgumentParser(description="Träna schack-NN från lokal Parquet-databas (Maraton-läge).")
    p.add_argument("--db-path", type=str, default=normal_databas, help="Sökväg till databasen.")
    p.add_argument("--model-path", type=str, default="chess_nnue.pth")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Mapp för sparade modeller.")
    
    p.add_argument("--epochs", type=int, default=109, help="Antal gånger nätverket ska se hela databasen.")
    
    # ÄNDRING 1: Höj batch-size till 1024 (eller 2048 om datorn klarar det) 
    # för att minska overhead och göra träningen snabbare per miljon rader.
    p.add_argument("--batch-size", type=int, default=1024, help="Antal ställningar per träningssteg.")
    
    p.add_argument("--lr", type=float, default=2e-4, help="Inlärningshastighet.")
    
    # ÄNDRING 2: Spara checkpoint mycket mer sällan (t.ex. var 25 000:e batch)
    # så du inte skapar tusentals filer som fyller disken under 20 timmar.
    p.add_argument("--checkpoint-every-batches", type=int, default=25000, help="Spara checkpoint var N:e batch.")
    
    return p.parse_args()

def get_target_value(cp: float | None, mate: float | None, clip: float = 10.0) -> float:
    """
    Konverterar cp eller mate till ett värde mellan -10.0 och 10.0 (pawns).
    Hanterar NaN och None (Eftersom NaN != NaN i Python, kollar vi mate == mate).
    """
    if mate is not None and mate == mate:  # Betyder att det finns ett giltigt mate-värde
        return clip if mate > 0 else -clip
    if cp is not None and cp == cp:        # Betyder att det finns ett giltigt cp-värde
        return max(-clip, min(clip, cp / 100.0))
    return 0.0


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Använder device: {device}")

# Ladda modellen och optimizers
    model = ChessEvaluatorNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500_000, eta_min=1e-5)
    
    load_model(model, args.model_path, device=device, optimizer=optimizer, scheduler=scheduler)
    model.train()
    
    loss_fn = nn.SmoothL1Loss()

    print(f"[train] Öppnar databas: {args.db_path}...")
    dataset = ds.dataset(args.db_path)

    log_file_path = "train_log.csv"
    start_batch = 0
    file_exists = os.path.exists(log_file_path)
    # ... (läser in start_batch från logg) ...

    # FLYTTAT HIT: batch_count sätts en gång utanför epokloopen
    batch_count = start_batch

    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\n=== Startar epok {epoch}/{args.epochs} ===")
            
            total_loss = 0.0
            t0 = time.time()
            
            X_list, y_list = [], []
            
            # Strömma data genom PyArrow (läser filer i stora block)
            for arrow_batch in dataset.to_batches():
                data_dict = arrow_batch.to_pydict()
                fens = data_dict['fen']
                cps = data_dict['cp']
                mates = data_dict['mate']
                
                for i in range(len(fens)):
                    board = chess.Board(fens[i])
                    X_list.append(board_to_tensor(board))
                    y_list.append(get_target_value(cps[i], mates[i]))
                    
                    # När vi har fyllt en mini-batch, skickar vi den till neurala nätverket
                    if len(X_list) == args.batch_size:
                        xb = torch.stack(X_list).to(device)
                        yb = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(device)
                        
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        
                        total_loss += loss.item()
                        batch_count += 1
                        X_list, y_list = [], []  # Återställ listorna
                        
                        # Skriv ut framsteg var 100:e batch
                        if batch_count % 100 == 0:
                            avg_loss = total_loss / 100
                            print(f"[train] Epok {epoch} | Batch {batch_count} | Snitt-loss: {avg_loss:.4f} | Tid: {time.time()-t0:.1f}s")
                            
                            # Spara till CSV i append-läge (mode="a") så filen inte rensas
                            with open(log_file_path, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow([epoch, batch_count, avg_loss])
                                
                            total_loss = 0.0
                            t0 = time.time()
                            
                        # Spara en checkpoint med jämna mellanrum
# Spara en checkpoint med jämna mellanrum
                        if batch_count % args.checkpoint_every_batches == 0:
                            ckpt_path = os.path.join(args.checkpoint_dir, f"chess_nnue_ep{epoch}_b{batch_count}.pth")
                            save_model(model, ckpt_path, optimizer=optimizer, scheduler=scheduler)
                            print(f"[train] Sparade checkpoint: {ckpt_path}")
                
            # Om det finns ställningar kvar i slutet av epoken som inte fyllde en hel batch
            if len(X_list) > 0:
                xb = torch.stack(X_list).to(device)
                yb = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_count += 1

            # Spara den senaste versionen när hela epoken är genomgången
            save_model(model, args.model_path)
            print(f"[train] Epok {epoch} klar. Huvudmodell sparad till {args.model_path}.")
            
    except KeyboardInterrupt:
        print("\n[train] Avbrutet av användaren (Ctrl+C). Sparar tränade vikter innan avslut...")
        
    finally:
        # Detta körs oavsett vad som händer (även vid KeyboardInterrupt)
        save_model(model, args.model_path, optimizer=optimizer, scheduler=scheduler)
        print(f"[train] Klart. Modellen har sparats slutgiltigt till '{args.model_path}'.")


if __name__ == "__main__":
    main()