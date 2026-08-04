# Schack-NN mot Stockfish (knowledge distillation)

Ett litet, komplett projekt: en PyTorch-utvärderare som tränas genom att "kopiera"
Stockfish, en negamax-sökning som faktiskt använder nätverket för att välja drag,
ett testläge med FEN/PGN-loggning, och ett benchmark-läge för att mäta framsteg
objektivt över tid.

## Filer

| Fil | Syfte |
|---|---|
| `nn_model.py` | `ChessEvaluatorNN`, `board_to_tensor`, spara/ladda vikter |
| `search.py` | Negamax + alpha-beta + quiescence, drivs av nätverket |
| `replay_buffer.py` | Disk-backad datamängd av `(ställning, Stockfish-facit)` |
| `train.py` | Träningsloop: bot vs Stockfish → samla data → träna |
| `test_selfplay.py` | Självspel + FEN/CSV-loggning + PGN-export |
| `benchmark.py` | Fast motstånd (Stockfish, fast Skill Level) → vinst/remi/förlust |
| `chess_nnue.pth` | Sparade vikter (skapas automatiskt vid första träning) |
| `replay_buffer.pt` | Sparad träningsdata (växer/återanvänds mellan körningar) |

Allt är testat i denna sandbox (tensor-kodning, forward-pass, sökning, buffer,
save/load, PGN-export, hela tränings- och testloopen) med en riggad Stockfish-mock,
eftersom den riktiga `.exe`-filen bara finns på din Windows-maskin. Logiken kring
själva UCI-anropen (`engine.analyse`, `engine.play`, `engine.configure`) följer
`python-chess`s dokumenterade API exakt, men du bör köra ett litet första test
(`--games 2`) för att bekräfta att allt pratar rätt med just din Stockfish-version.

## Installation

```bash
pip install -r requirements.txt
```

Se till att `stockfish-windows-x86-64-avx2.exe` antingen ligger i samma mapp som
skripten, eller pekas ut explicit:

```bash
python train.py --stockfish-path stockfish-windows-x86-64-avx2.exe --games 50
```

(Ligger den redan i PATH räcker det med filnamnet, som i default-värdet.)

## 1. Träna

```bash
python train.py --games 200 --game-depth 2 --stockfish-eval-depth 12 --skill-level 8
```

Vad som händer:
- Botten spelar en färg (styrd av `search.py` + nätverket), Stockfish på
  `--skill-level` (0–20) spelar den andra, för realistisk variation.
- Efter *varje halvdrag* frågas Stockfish om ett facit på `--stockfish-eval-depth`
  (djupare = bättre men långsammare facit).
- Data läggs i `replay_buffer.pt`, som växer över körningar (och begränsas av
  `--max-buffer-size` genom att slumpvis behålla ett urval om den blir för stor).
- Var `--train-every`:e parti tränas nätverket ett par epoker på bufferten
  (`--epochs`, `--batch-size`, `--lr`), med en liten valideringsdel så du ser om
  `val_loss` faktiskt sjunker (inte bara `train_loss`).
- Modellen sparas kontinuerligt till `chess_nnue.pth`, plus namngivna
  checkpoints (`chess_nnue_ep_25.pth` osv) var `--checkpoint-every`:e parti.
- Ctrl+C när som helst → sparar innan avslut.

Kör skriptet igen senare med samma `--model-path`/`--buffer-path` så plockar det
upp där du slutade (vikter + data laddas automatiskt).

**Prestanda-tips:** varje sökt drag kostar en eller flera forward-pass genom
nätverket i Python, vilket är betydligt dyrare än en handskriven eval-funktion.
Håll `--game-depth` lågt (1–2) tills du vill optimera vidare (t.ex. batcha
evalueringar av flera bladnoder samtidigt, eller flytta hot-path till
NumPy/Cython/PyPy — precis den typen av optimering du redan jobbat med i dina
andra projekt).

## 2. Testa (självspel + loggning)

```bash
python test_selfplay.py --model-path chess_nnue.pth --games 5 --depth 2
```

- Skriver ut brädet i terminalen efter varje drag (`--no-print-board` för att
  slå av).
- Loggar varje halvdrag till `test_fen_log.csv`: parti, dragnummer, sida,
  FEN, nätverkets utvärdering, drag i SAN och UCI.
- Sparar alla partier till `self_play_tests.pgn` — öppningsbar i Arena/lichess.

Vill du jämföra två olika checkpoints mot varandra (t.ex. dagens modell mot
förra veckans):

```bash
python test_selfplay.py --model-path chess_nnue.pth --model-b-path chess_nnue_ep_25.pth
```

## 3. Bli bättre (mät faktiska framsteg)

Sjunkande träningsförlust betyder inte automatiskt en starkare bot — nätverket
kan bli bra på att gissa Stockfishs siffror utan att spelet faktiskt förbättras.
`benchmark.py` ger ett objektivt, jämförbart mått: samma motstånd, samma styrka,
varje gång.

```bash
python benchmark.py --model-path chess_nnue.pth --games 20 --skill-level 5
```

Kör detta med **samma** `--skill-level` efter varje träningsomgång och jämför
poängen (vinster + 0.5×remi) rakt av. En naturlig loop:

1. `train.py --games 100` (bygg buffer, träna)
2. `benchmark.py --games 20` (mät)
3. Justera hyperparametrar om det stagnerar — t.ex. höj `--stockfish-eval-depth`
   för bättre facit, eller `--skill-level` för svårare motstånd när botten börjar
   vinna konsekvent (curriculum learning)
4. Upprepa

## Kända begränsningar / naturliga nästa steg

- **Feature-representationen** är en enkel platt 775-dimensionell vektor
  (12 pjärsplan × 64 rutor + 7 extra-features), inte riktig HalfKP/NNUE med
  kung-relativa features och inkrementella uppdateringar. Fungerar bra som
  start, men en HalfKP-representation skulle ge både snabbare inferens och
  bättre spelstyrka — en bra kandidat för nästa optimeringsrunda.
- **Sökningen** är enkel alpha-beta + grund quiescence, ingen iterative
  deepening, transposition table eller move-ordering utöver MVV-LVA. Går att
  koppla ihop med dina tidigare lärdomar från minimax-arbetet (transposition
  cache, mattpoäng-djustering, etc.) om du vill slå ihop projekten.
- **Remi-hantering** i `evaluate()`/sökningen kollar bara patt och otillräckligt
  material (för prestanda i hot path) — 50-dragsregeln och trefaldig upprepning
  hanteras på partinivå (`claim_draw=True`) i spel-looparna, inte inne i
  sökrekursionen.
