import chromadb
from mistralai import Mistral

# 1. CONFIGURATION
api_key = "ugvXJKDDMHOfZ66NrXHeePsYuU4KUpUn"  # Collez votre clé ici
client_mistral = Mistral(api_key=api_key)
chroma_client = chromadb.Client() # Mode mémoire (s'efface quand on ferme le script)

# On crée une collection temporaire pour ce test
collection = chroma_client.create_collection(name="demo_rag")

# 2. VOS DONNÉES (La "Connaissance")
# Imaginons un univers de Sci-Fi simple
textes_de_reference = [
    "La planète Zylos est recouverte à 90% d'océans violets acides.",
    "Les habitants de Zylos s'appellent les Zyliens et respirent par la peau.",
    "La capitale de Zylos est une cité flottante nommée Aquaria.",
    "Le plat préféré des Zyliens est la méduse électrique cristallisée."
]

# On ajoute ces textes dans la base de données
# ChromaDB transforme automatiquement le texte en vecteurs (embeddings) ici
collection.add(
    documents=textes_de_reference,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

print("✅ Base de connaissances chargée (4 documents).")

# 3. LA BOUCLE DE QUESTION
while True:
    question_utilisateur = input("\nPosez une question sur Zylos (ou 'quit' pour sortir) : ")
    
    if question_utilisateur.lower() == 'quit':
        break

    # ÉTAPE A : RECHERCHE (Retrieval)
    # On demande à la base : "Trouve le document le plus proche de cette question"
    resultats = collection.query(
        query_texts=[question_utilisateur],
        n_results=1  # On veut juste le TOP 1 résultat le plus pertinent
    )
    
    contexte_trouve = resultats['documents'][0][0]
    print(f"   (Info trouvée dans la base : '{contexte_trouve}')")

    # ÉTAPE B : GÉNÉRATION (Generation)
    # On construit le prompt final pour l'IA
    prompt = f"""
    Utilise l'information suivante pour répondre à la question.
    Information : {contexte_trouve}
    
    Question : {question_utilisateur}
    """

    # On envoie le tout à Mistral
    reponse = client_mistral.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"🤖 Réponse : {reponse.choices[0].message.content}")