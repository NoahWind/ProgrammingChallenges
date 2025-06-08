from mpmath import mp, mpf, sqrt, fac
import os
import time

# Starta med rimlig initialprecision
mp.dps = 30000

# Fil att spara till
output_file = "where are you in pi/pi_progress.txt"

# Chudnovsky-term
def chudnovsky_term(k):
    M = fac(6*k) / (fac(3*k) * fac(k)**3)
    L = 13591409 + 545140134*k
    X = (-262537412640768000)**k
    return mpf(M * L) / X

# Beräkna pi i oändlighet
def run_pi():
    sum_ = mpf(0)
    k = 0

    try:
        while True:
            term = chudnovsky_term(k)
            sum_ += term
            approx_pi = (426880 * sqrt(10005)) / sum_

            # Öka precision lite över aktuell nivå
            digits_now = len(str(approx_pi).replace('.', ''))
            mp.dps = digits_now + 10000

            # Visa status ibland (valfritt)
            if k % 5 == 0:
                print(f"{digits_now} siffror beräknade... Iteration {k}")

            k += 1

    except KeyboardInterrupt:
        print(f"\nCtrl+C mottaget. Sparar resultat till '{output_file}'...")
        with open(output_file, "w") as f:
            f.write(str(approx_pi) + "\n")
        print(f"✅ Sparade {digits_now} siffror av π.")

if __name__ == "__main__":
    run_pi()
