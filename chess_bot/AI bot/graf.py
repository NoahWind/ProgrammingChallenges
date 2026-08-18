import pandas as pd
import matplotlib.pyplot as plt

def main():
    log_file_path = "train_log.csv"
    
    try:
        # Läs in filen rad för rad
        df = pd.read_csv(log_file_path, header=None, names=["epoch", "batch", "loss"])
    except FileNotFoundError:
        print(f"Hittade ingen loggfil vid '{log_file_path}'. Kör träningen först!")
        return

    # Tvångskonvertera loss-kolumnen till siffror
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    
    # Ta bort rader som inte gick att konvertera
    df = df.dropna(subset=["loss"])

    if df.empty:
        print("Loggfilen är tom eller saknar giltig numerisk data!")
        return

    # Skapa grafen
    plt.figure(figsize=(10, 5))
    
    # Använd radindex (0, 1, 2, ...) som x-axel för att läsa av sekventiellt uppifrån och ner
    plt.plot(df.index, df["loss"], label="Snitt-loss", color="blue", alpha=0.7)
    
    # Glidande medelvärde baserat på den sekventiella ordningen
    plt.plot(df.index, df["loss"].rolling(window=10).mean(), label="Trend (Glidande medelvärde)", color="red", linewidth=2)

    plt.xlabel("Steg / Rad (uppifrån och ner)")
    plt.ylabel("Loss (Felmarginal)")
    plt.title("Träningsframsteg – Sekventiell avläsning")
    plt.legend()
    plt.grid(True)
    
    print("Visar graf...")
    plt.show()

if __name__ == "__main__":
    main()