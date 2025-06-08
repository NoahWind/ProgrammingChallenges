from datasets import load_dataset

ds = load_dataset("RZ412/PokerBench")

# Visa första exemplet
print(ds["train"][0])
