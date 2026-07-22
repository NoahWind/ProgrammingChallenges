# -*- coding: utf-8 -*-
"""
OBS - anpassad till nya chess_rules.py (ersätter get_leagle_moves.py):

- get_all_legal_moves(color, board, state) tar nu ett GameState-objekt
  istället för en fen-sträng och globala variabler.
- Egna is_check/is_mate-implementationerna är borttagna - chess_rules har
  redan snabba, korrekta motsvarigheter (is_check, is_checkmate,
  is_stalemate) som INTE scannar brädet i onödan.
- Drag utförs nu med apply_move(board, state, start, end, promotion) som
  hanterar rockad (flyttar tornet), en passant (tar bort rätt bonde) och
  uppdaterar alla rättighets-flaggor korrekt. Den gamla manuella
  board[er][ec] = board[sr][sc]-koden gjorde INGET av detta, vilket med
  stor sannolikhet var boven bakom brädkorruptionen du fick.
- FEN-parsern läser nu även rockad-fältet och en passant-fältet (fält 3
  och 4), inte bara pjäsplacering och vem som har draget - annars hade
  GameState alltid startat med "alla rockad-rättigheter kvar" oavsett
  vad FEN faktiskt sa.

VIKTIGT: get_best_move i rating.py måste anpassas till samma API
(ta emot `state`, skicka det vidare till get_all_legal_moves, och helst
använda apply_move/undo_move för make/unmake i sökträdet istället för
att djupkopiera brädet vid varje nod - annars blir djup 15 mycket segt).
Skicka rating.py så hjälper jag till att uppdatera den också.
"""

import time
import datetime

from chess_rules import (
    GameState,
    get_all_legal_moves,
    is_check,
    apply_move,
    undo_move, 
    get_position_hash
)
from rating import get_best_move


SEARCH_DEPTH = 2000000
MAX_MOVES = 3000  # säkerhetsspärr - ingen 50-dragsregel/remi-detektion finns än,
                  # så utan en gräns kan partiet i teorin loopa i all oändlighet

PIECE_LETTERS = {
    '♚': 'K', '♛': 'Q', '♜': 'R', '♝': 'B', '♞': 'N', '♟': '',
    '♔': 'K', '♕': 'Q', '♖': 'R', '♗': 'B', '♘': 'N', '♙': '',
}

PIECE_NAMES_SV = {
    'K': 'Kung', 'Q': 'Dam', 'R': 'Torn', 'B': 'Löpare', 'N': 'Springare', '': 'Bonde',
}


def square_to_algebraic(r, c):
    """Konverterar (rad, kolumn) till schackruta, t.ex. (6, 0) -> 'a2'."""
    file = chr(ord('a') + c)
    rank = str(8 - r)
    return f"{file}{rank}"


def move_to_notation(board, move):
    """
    Konverterar ett drag ((sr, sc), (er, ec)) till läsbar schacknotation.
    ENDAST till för utskrift - drag skickas fortfarande som rad/kolumn-tupler
    till get_all_legal_moves/get_best_move precis som innan.
    """
    (sr, sc), (er, ec) = move
    piece = board[sr][sc]
    piece_letter = PIECE_LETTERS.get(piece, '')

    # Rockad: kungen flyttar två steg i sidled
    if piece in ('♔', '♚') and abs(ec - sc) == 2:
        return "O-O" if ec > sc else "O-O-O"

    start_sq = square_to_algebraic(sr, sc)
    end_sq = square_to_algebraic(er, ec)
    is_capture = board[er][ec] != ' '
    separator = 'x' if is_capture else '-'
    return f"{piece_letter}{start_sq}{separator}{end_sq}"


def move_description(board, move):
    """Fylligare, läsbar beskrivning av draget - t.ex. 'Dam d3 → d5 (slag)'."""
    (sr, sc), (er, ec) = move
    piece = board[sr][sc]
    letter = PIECE_LETTERS.get(piece, '')
    name = PIECE_NAMES_SV.get(letter, 'Bonde')

    if piece in ('♔', '♚') and abs(ec - sc) == 2:
        return "Kort rockad (O-O)" if ec > sc else "Lång rockad (O-O-O)"

    start_sq = square_to_algebraic(sr, sc)
    end_sq = square_to_algebraic(er, ec)
    is_capture = board[er][ec] != ' '
    tail = " (slag)" if is_capture else ""
    return f"{name} {start_sq} → {end_sq}{tail}"


def make_board_from_fen(fen):
    piece_map = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
    }

    board = [[' ' for _ in range(8)] for _ in range(8)]
    piece_placement = fen.split(' ')[0]
    rows = piece_placement.split('/')

    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board[i][col] = piece_map.get(char, char)
                col += 1
    return board


def parse_fen(fen):
    """
    Läser ut bräde, vem som har draget, OCH rockad-/en passant-rättigheter
    ur en FEN-sträng och bygger ett GameState som stämmer med FEN:en
    (istället för att bara anta att allt är tillåtet från start).
    """
    fields = fen.split()
    board = make_board_from_fen(fen)

    color = 'white' if len(fields) > 1 and fields[1] == 'w' else 'black'
    castling = fields[2] if len(fields) > 2 else '-'
    en_passant_field = fields[3] if len(fields) > 3 else '-'

    state = GameState()
    if 'K' not in castling:
        state.rook_h1_moved = True
    if 'Q' not in castling:
        state.rook_a1_moved = True
    if 'k' not in castling:
        state.rook_h8_moved = True
    if 'q' not in castling:
        state.rook_a8_moved = True

    if en_passant_field != '-':
        col = ord(en_passant_field[0]) - ord('a')
        row = 8 - int(en_passant_field[1])
        state.en_passant_target = (row, col)
    
    state.init_pieces(board)

    return board, color, state


def print_board(board, last_move=None):
    """
    Skriver ut brädet. Om last_move ges (samma format som move_to_notation
    förväntar sig, dvs ((sr,sc),(er,ec))) markeras rutan pjäsen kom IFRÅN
    med '·' så man ser varifrån den flyttade, inte bara vart den hamnade.
    """
    highlight_from = last_move[0] if last_move else None

    print("  a b c d e f g h")
    for i in range(8):
        print(8 - i, end=' ')
        for j in range(8):
            cell = board[i][j]
            if highlight_from == (i, j) and cell == ' ':
                cell = '·'
            print(cell, end=' ')
        print(8 - i)
    print("  a b c d e f g h")

def move_to_san(board, state, move, current_color, is_check_res, is_mate_res):
    (sr, sc), (er, ec) = move
    piece = board[sr][sc]
    letter = PIECE_LETTERS.get(piece, '')
    
    # Identifiera slag (inklusive En Passant)
    is_capture = board[er][ec] != ' '
    if piece in ('♙', '♟') and sc != ec and board[er][ec] == ' ':
        is_capture = True

    # Rockad
    if piece in ('♔', '♚') and abs(ec - sc) == 2:
        san = "O-O" if ec > sc else "O-O-O"
    else:
        end_sq = square_to_algebraic(er, ec)
        if piece in ('♙', '♟'):
            san = f"{chr(ord('a') + sc)}x{end_sq}" if is_capture else end_sq
        else:
            # Undvik tvetydighet (t.ex. om två springare kan gå till samma ruta)
            legal_moves = get_all_legal_moves(current_color, board, state)
            similar_pieces = [m for m in legal_moves 
                              if m != move and m[1] == (er, ec) and board[m[0][0]][m[0][1]] == piece]
            
            disambig = ""
            if similar_pieces:
                if all(m[0][1] != sc for m in similar_pieces):
                    disambig = chr(ord('a') + sc)
                elif all(m[0][0] != sr for m in similar_pieces):
                    disambig = str(8 - sr)
                else:
                    disambig = square_to_algebraic(sr, sc)
                    
            capture_mark = "x" if is_capture else ""
            san = f"{letter}{disambig}{capture_mark}{end_sq}"
            
    # Promotion (Motorn byter automatiskt till dam)
    if piece in ('♙', '♟') and (er == 0 or er == 7):
        san += "=Q"
        
    # Schack och Matt-markeringar
    if is_mate_res:
        san += "#"
    elif is_check_res:
        san += "+"
        
    return san


def play_game_from_FEN(fen):
    board, current_color, state = parse_fen(fen)
    print_board(board)
    history = {}  # Här sparar vi spelets faktiska historik
    
    pgn_move_list = [] 
    game_result = "*"  

    for move_number in range(1, MAX_MOVES + 1):
        # --- 1. KOLLA 3-FALDIG UPPREPNING FÖRST ---
        board_hash = get_position_hash(board, current_color, state)
        history[board_hash] = history.get(board_hash, 0) + 1
        
        # Kräver att ställningen skett 3 gånger OCH att det inte är ett dött slutspel med stor materialfördel
        if history[board_hash] >= 4 and abs(state.material_score) < 5:
            print("Remi genom 3-faldig upprepning!")
            game_result = "1/2-1/2"
            break

        legal_moves = get_all_legal_moves(current_color, board, state)
        
        # --- 50-DRAGSREGELN ---
        if state.half_move_clock >= 100:
            print("Remi genom 50-dragsregeln!")
            game_result = "1/2-1/2"
            break
            
        # --- MATT / PATT ---
        if not legal_moves:
            if is_check(board, current_color, state):
                print(f"\n{current_color.capitalize()} är schackmatt. Partiet är slut.")
                game_result = "0-1" if current_color == 'white' else "1-0"
            else:
                print(f"\n{current_color.capitalize()} har inga lagliga drag (patt). Remi.")
                game_result = "1/2-1/2"
            break

        print(f"\nDrag {move_number}: {current_color} tänker (djup {SEARCH_DEPTH})...")
        move_start = time.time()
        
        # VIKTIGT: Skicka med den rena spelets historik med .copy() 
        # så sökmotors fiktiva tankar inte kladdar här!
        best_move = get_best_move(board, SEARCH_DEPTH, current_color, state, history.copy())
        move_end = time.time()
        
        if not best_move:
            print(f"{current_color.capitalize()} hittade inget drag. Partiet är slut.")
            break

        # --- SIMULERA OCH ÖVERSÄTT TILL PGN ---
        record = apply_move(board, state, best_move[0], best_move[1])
        
        # Kontrollera om draget ledde till schack/matt för motståndaren
        next_color = 'black' if current_color == 'white' else 'white'
        legal_next_moves = get_all_legal_moves(next_color, board, state)
        is_check_res = is_check(board, next_color, state)
        is_mate_res = is_check_res and not legal_next_moves
        
        # Ångra draget temporärt för att pjäserna ska stå rätt när vi bygger SAN-notationen
        undo_move(board, state, record)
        san_move = move_to_san(board, state, best_move, current_color, is_check_res, is_mate_res)
        
        # Formatera PGN-texten (T.ex. "1. e4" för vit, "e5" för svart)
        full_move_number = (move_number + 1) // 2
        if current_color == 'white':
            pgn_move_list.append(f"{full_move_number}. {san_move}")
        else:
            pgn_move_list.append(san_move)

        # Gör draget på riktigt
        apply_move(board, state, best_move[0], best_move[1])

        duration_per_move = move_end - move_start
        print(f"  -> {move_description(board, best_move)}  [{san_move}]")
        print(f"  -> Dragtid: {duration_per_move:.2f} sekunder")

        print_board(board, last_move=best_move)

        current_color = next_color
    else:
        print(f"\nStoppade efter {MAX_MOVES} drag utan matt/patt (säkerhetsspärr).")

    # --- SKAPA OCH SPARA PGN-FILEN ---
    date_str = datetime.datetime.now().strftime("%Y.%m.%d")
    pgn_string = (
        f'[Event "Engine Match"]\n'
        f'[Site "Local"]\n'
        f'[Date "{date_str}"]\n'
        f'[Round "1"]\n'
        f'[White "White Engine"]\n'
        f'[Black "Black Engine"]\n'
        f'[Result "{game_result}"]\n\n'
        f'{" ".join(pgn_move_list)} {game_result}\n'
    )
    
    with open("match.pgn", "w", encoding="utf-8") as f:
        f.write(pgn_string)
    
    print("\n---------------------------------------------------")
    print("Spelet är över. Matchen har sparats till 'match.pgn'!")

def main():
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    play_game_from_FEN(fen_start)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nTotal tid: {end - start:.2f} sekunder")