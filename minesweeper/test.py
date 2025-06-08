import chromadb

# Starta en PersistentClient i ny stil
client = chromadb.PersistentClient(path="./chroma_db")

# Hämta eller skapa en collection
collection = client.get_or_create_collection("min_collection")

# Lägg till dokument
collection.add(
    documents=["Hej världen!", "Kod är kung!"],
    metadatas=[{"typ": "hälsning"}, {"typ": "skämt"}],
    ids=["1", "2"]
)

# Sök
result = collection.query(query_texts=["Hej"], n_results=2)
print(result)
