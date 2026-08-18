"""
play_bot.py
-----------
Spela mot din tränade schack-AI i ett Tkinter-gränssnitt!
Stockfish körs vid sidan av som en kommentator.
"""

import tkinter as tk
from tkinter import messagebox
import chess
import chess.engine
import torch
import os

# Importera din modells arkitektur
from nn_model import ChessEvaluatorNN, board_to_tensor, load_model

# --- Inställningar ---
SQUARE_SIZE = 64
BOARD_COLORS = ["#eed8a1", "#b58863"] # Ljusa och mörka fält
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "chess_nnue.pth"

# Unicode för schackpjäser
PIECE_SYMBOLS = {
    chess.Piece(chess.PAWN, chess.WHITE): "♙",
    chess.Piece(chess.KNIGHT, chess.WHITE): "♘",
    chess.Piece(chess.BISHOP, chess.WHITE): "♗",
    chess.Piece(chess.ROOK, chess.WHITE): "♖",
    chess.Piece(chess.QUEEN, chess.WHITE): "♕",
    chess.Piece(chess.KING, chess.WHITE): "♔",
    
    chess.Piece(chess.PAWN, chess.BLACK): "♟",
    chess.Piece(chess.KNIGHT, chess.BLACK): "♞",
    chess.Piece(chess.BISHOP, chess.BLACK): "♝",
    chess.Piece(chess.ROOK, chess.BLACK): "♜",
    chess.Piece(chess.QUEEN, chess.BLACK): "♛",
    chess.Piece(chess.KING, chess.BLACK): "♚"
}

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spela mot NNUE Boten")
        
        self.board = chess.Board()
        self.selected_square = None
        
        # --- Ladda Neurala Nätverket ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChessEvaluatorNN().to(self.device)
        if os.path.exists(MODEL_PATH):
            load_model(self.model, MODEL_PATH, device=self.device)
            self.model.eval()
            print(f"[AI] Modell laddad från {MODEL_PATH} på {self.device}.")
        else:
            messagebox.showerror("Fel", f"Hittade inte modellen: {MODEL_PATH}")
            self.root.destroy()
            return

        # --- Ladda Stockfish ---
        if os.path.exists(STOCKFISH_PATH):
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            print("[Stockfish] Motor startad.")
        else:
            self.engine = None
            print("[Stockfish] Hittades inte. Analys inaktiverad.")

        # --- Gränssnitt ---
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.sf_label = tk.Label(self.info_frame, text="Stockfish: Väntar...", font=("Arial", 12, "bold"), fg="blue")
        self.sf_label.pack(side=tk.LEFT)
        
        self.bot_label = tk.Label(self.info_frame, text="Bot (Svart): Väntar...", font=("Arial", 12))
        self.bot_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, width=SQUARE_SIZE*8, height=SQUARE_SIZE*8)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()
        self.update_analysis()

    def draw_board(self):
        self.canvas.delete("all")
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Beräkna koordinater för Tkinter (y är inverterat)
            x0 = file * SQUARE_SIZE
            y0 = (7 - rank) * SQUARE_SIZE
            x1 = x0 + SQUARE_SIZE
            y1 = y0 + SQUARE_SIZE
            
            color = BOARD_COLORS[(file + rank) % 2 == 0]
            if self.selected_square == square:
                color = "#a1c96a" # Markera vald ruta med grönt
                
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            
            piece = self.board.piece_at(square)
            if piece:
                symbol = PIECE_SYMBOLS.get(piece, "")
                self.canvas.create_text(x0 + SQUARE_SIZE//2, y0 + SQUARE_SIZE//2, 
                                        text=symbol, font=("Arial", int(SQUARE_SIZE*0.6)))

    def on_click(self, event):
        if self.board.turn != chess.WHITE:
            return # Det är botens tur
            
        file = event.x // SQUARE_SIZE
        rank = 7 - (event.y // SQUARE_SIZE)
        clicked_square = chess.square(file, rank)
        
        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = clicked_square
                self.draw_board()
        else:
            # Försök göra ett drag
            move = chess.Move(self.selected_square, clicked_square)
            
            # Hantera bondeförvandling (autodam)
            if self.board.piece_at(self.selected_square) and self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) == 7:
                    move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                self.update_analysis()
                
                # Check för schackmatt
                if not self.check_game_over():
                    self.root.after(100, self.bot_move)
            else:
                # Avmarkera om draget var ogiltigt
                self.selected_square = None
                self.draw_board()

    def bot_move(self):
        self.bot_label.config(text="Bot tänker (djup 2)...")
        self.root.update()

        best_move = None
        best_eval = float('inf') # Svart vill minimera poängen
        alpha = float('-inf')
        beta = float('inf')
        
        # Sökdjup (Ändra denna om du vill ha djupare/grundare, t.ex. 2 eller 3)
        SEARCH_DEPTH = 5

        # Alpha-Beta Minimax för Svart (som vill minimera poängen)
        for move in self.board.legal_moves:
            self.board.push(move)
            # Nästa drag är Vits tur (maximera), så vi skickar med maximizing=True
            eval_score = self.alpha_beta(SEARCH_DEPTH - 1, alpha, beta, maximizing_player=True)
            self.board.pop()
            
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move

        if best_move:
            self.board.push(best_move)
            self.bot_label.config(text=f"Bot Eval: {best_eval:.2f} CP")
            self.draw_board()
            self.update_analysis()
            self.check_game_over()

    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        """Rekursiv Minimax-funktion med Alpha-Beta-pruning."""
        if depth == 0 or self.board.is_game_over():
            # Utvärdera ställningen med ditt neurala nätverk
            tensor_board = board_to_tensor(self.board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model(tensor_board).item()

        if maximizing_player: # Vit (vill ha högt värde)
            max_eval = float('-inf')
            for move in self.board.legal_moves:
                self.board.push(move)
                eval = self.alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Pruning
            return max_eval
        else: # Svart (vill ha lågt värde)
            min_eval = float('inf')
            for move in self.board.legal_moves:
                self.board.push(move)
                eval = self.alpha_beta(depth - 1, alpha, beta, True)
                self.board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Pruning
            return min_eval

        if best_move:
            self.board.push(best_move)
            self.bot_label.config(text=f"Bot Eval: {best_eval:.2f} CP")
            self.draw_board()
            self.update_analysis()
            self.check_game_over()

    def update_analysis(self):
        if self.engine:
            try:
                # Stockfish analyserar blixtsnabbt
                info = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
                score = info["score"].white()
                
                if score.is_mate():
                    txt = f"Stockfish: Matt i {abs(score.mate())}"
                else:
                    txt = f"Stockfish: {score.score(mate_score=10000)/100:.2f} CP"
                    
                self.sf_label.config(text=txt)
            except Exception as e:
                self.sf_label.config(text="Stockfish: Fel")
                
    def check_game_over(self):
        if self.board.is_checkmate():
            vinnare = "Vit" if self.board.turn == chess.BLACK else "Svart"
            messagebox.showinfo("Schackmatt!", f"{vinnare} vann!")
            return True
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            messagebox.showinfo("Remi!", "Spelet slutade oavgjort.")
            return True
        return False

    def on_closing(self):
        if self.engine:
            self.engine.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()