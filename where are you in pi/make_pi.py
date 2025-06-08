from mpmath import mp, mpf, sqrt, fac
import os
import traceback
import time

# Starta med en måttlig precision
mp.dps = 300000  # Högre startprecision

def chudnovsky_term(k):
    try:
        M = fac(6*k) / (fac(3*k) * fac(k)**3)
        L = 13591409 + 545140134*k
        X = (-262537412640768000)**k
        return mpf(M * L) / X
    except OverflowError:
        # Öka precisionen om vi får overflow
        mp.dps *= 2
        print(f"\nÖverflödesfel upptäckt. Ökar precision till {mp.dps}")
        return chudnovsky_term(k)  # Försök igen

def run_pi():
    sum_ = mpf(0)
    k = 0
    best_prefix = ""
    last_progress_time = time.time()
    last_digit_count = 0
    stalled_count = 0
    
    try:
        while True:
            try:
                # Beräkna termen och lägg till i summan
                term = chudnovsky_term(k)
                sum_ += term
                
                # Beräkna pi-approximationen
                approx_pi = (426880 * sqrt(10005)) / sum_
                
                # Hämta den "sanna" pi med nuvarande precision
                real_pi = mp.pi
                
                # Jämför med riktiga pi
                a_str = str(approx_pi)
                b_str = str(real_pi)
                i = 0
                while i < min(len(a_str), len(b_str)) and a_str[i] == b_str[i]:
                    i += 1
                
                current_prefix = a_str[:i]
                correct_digits = len(current_prefix.replace('.', ''))
                
                # Kontrollera om vi gör framsteg
                current_time = time.time()
                if correct_digits > len(best_prefix.replace(".", "")):
                    best_prefix = current_prefix
                    print(f"{correct_digits} siffror säkra: {best_prefix[:50]}...")
                    last_progress_time = current_time
                    last_digit_count = correct_digits
                    stalled_count = 0
                elif current_time - last_progress_time > 5:  # 30 sekunder utan framsteg
                    stalled_count += 1
                    print(f"Inga nya framsteg på 5 sekunder. Ökar precisionen ({stalled_count})")
                    mp.dps *= 5
                    print(f"Ny precision: {mp.dps}")
                    real_pi = mp.pi  # Uppdatera pi med den nya precisionen
                    last_progress_time = current_time
                    
                    # Om vi fortsätter att se samma antal siffror efter flera ökningar
                    if stalled_count >= 3 and correct_digits == last_digit_count:
                        print("Fastnat vid samma antal siffror efter flera precisionsökningar.")
                        print("Provar en större ökning av precisionen...")
                        mp.dps *= 10  # Mer aggressiv ökning
                        real_pi = mp.pi
                    
                # Öka precisionen om vi närmar oss gränsen
                if correct_digits > mp.dps * 0.7:
                    new_dps = mp.dps * 16
                    print(f"\nNärmar oss precisionsgräns. Ökar från {mp.dps} till {new_dps} decimaler")
                    mp.dps = new_dps
                    real_pi = mp.pi  # Uppdatera pi med den nya precisionen
                
                k += 1
                
            except (mp.NoConvergence, OverflowError):
                # Om beräkningen inte konvergerar eller överflödar, öka precisionen kraftigt
                new_dps = mp.dps * 4  # Mer aggressiv ökning vid fel
                print(f"\nBeräkningen misslyckades. Ökar precision från {mp.dps} till {new_dps} decimaler")
                mp.dps = new_dps
                real_pi = mp.pi  # Uppdatera pi med den nya precisionen
                continue

    except KeyboardInterrupt:
        # Hantera när användaren avbryter med Ctrl+C
        new_digits = len(best_prefix.replace(".", ""))
        print(f"\nBeräkning avbruten. Nådde {new_digits} siffror.")
        
        file_path = r"where are you in pi\pi.csv"
        old_digits = 0
        
        # Försök skapa katalogen om den inte existerar
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except:
            file_path = "pi.csv"  # Fallback till enklare sökväg vid problem
            print(f"Kunde inte skapa katalogstruktur. Använder {file_path} istället.")

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    old_pi = f.readline().strip()
                    old_digits = len(old_pi.replace(".", ""))
            except:
                print("Kunde inte läsa tidigare pi-fil.")
                old_digits = 0

        if new_digits > old_digits:
            try:
                with open(file_path, "w") as f:
                    f.write(f"{best_prefix}\n")
                print(f"\n✅ Nytt rekord! Sparade {new_digits} siffror av π.")
            except Exception as e:
                print(f"Kunde inte spara till fil: {str(e)}")
                # Försök spara till en enklare plats
                with open("pi_backup.txt", "w") as f:
                    f.write(f"{best_prefix}\n")
                print("Sparade till pi_backup.txt istället.")
        else:
            print(f"\n🚫 Inget sparat. Nuvarande ({new_digits}) var inte bättre än tidigare ({old_digits}).")
    
    except Exception as e:
        print(f"\nEtt fel uppstod: {str(e)}")
        print(traceback.format_exc())
        print(f"Sista framgångsrika beräkning nådde {len(best_prefix.replace('.', ''))} siffror.")
        
        # Försök spara det vi har i en nödfil
        try:
            with open("pi_emergency.txt", "w") as f:
                f.write(f"{best_prefix}\n")
            print(f"Sparade nödfil med {len(best_prefix.replace('.', ''))} siffror.")
        except:
            pass

if __name__ == "__main__":
    run_pi()