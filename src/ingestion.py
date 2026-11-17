from pathlib import Path
import hashlib
import re

import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOSSIER_PDF = BASE_DIR / "data" / "pdf"

DOSSIER_PDF.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_or_create_collection(
    name="entreprise_documents"
)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ==========================================
# 2. DÉCOUPAGE DU TEXTE (par phrases)
# ==========================================

def _couper_phrase_longue(phrase, taille):
    """Coupe une phrase trop longue sur les espaces (jamais au milieu d'un mot)."""
    morceaux, courant = [], ""

    for mot in phrase.split():
        if courant and len(courant) + len(mot) + 1 > taille:
            morceaux.append(courant)
            courant = mot
        else:
            courant = f"{courant} {mot}".strip()

    if courant:
        morceaux.append(courant)

    return morceaux


def decouper_texte(texte, taille=600, chevauchement=1):
    """
    Découpe en chunks d'environ `taille` caractères en respectant les
    phrases. `chevauchement` = nombre de phrases répétées entre 2 chunks.
    """
    if taille <= 0 or chevauchement < 0:
        raise ValueError("Paramètres de découpage invalides.")

    # Recoller les lignes cassées par le PDF
    texte = re.sub(r"\s*\n\s*", " ", texte).strip()

    if not texte:
        return []

    phrases = []

    for phrase in re.split(r"(?<=[.!?])\s+", texte):
        if len(phrase) > taille:
            phrases.extend(_couper_phrase_longue(phrase, taille))
        elif phrase.strip():
            phrases.append(phrase.strip())

    chunks, courant, longueur = [], [], 0

    for phrase in phrases:

        if courant and longueur + len(phrase) > taille:
            chunks.append(" ".join(courant))

            courant = courant[-chevauchement:] if chevauchement else []
            longueur = sum(len(p) + 1 for p in courant)

        courant.append(phrase)
        longueur += len(phrase) + 1

    if courant:
        chunks.append(" ".join(courant))

    return chunks


# ==========================================
# 3. INDEXER UN PDF
# ==========================================

def indexer_pdf(fichier_pdf):

    fichier_pdf = Path(fichier_pdf)

    ids = []
    textes = []
    metadonnees = []

    # Identifiant stable par nom de fichier
    hash_nom = hashlib.sha256(
        fichier_pdf.name.encode("utf-8")
    ).hexdigest()[:12]

    # Lire les pages du PDF
    with fitz.open(fichier_pdf) as document:

        for numero_page, page in enumerate(
            document,
            start=1
        ):

            texte_page = page.get_text()

            morceaux = decouper_texte(texte_page)

            for numero_chunk, morceau in enumerate(
                morceaux,
                start=1
            ):

                ids.append(
                    f"{hash_nom}_p{numero_page}_c{numero_chunk}"
                )

                textes.append(morceau)

                metadonnees.append({
                    "source": fichier_pdf.name,
                    "page": numero_page,
                    "numero_chunk": numero_chunk
                })

    if not textes:
        raise ValueError(
            "Aucun texte extractible trouvé dans ce PDF "
            "(document scanné ?)."
        )

    # Transformer les textes en embeddings
    embeddings = model.encode(textes).tolist()

    # Retirer les anciens chunks du même fichier
    anciens = collection.get(
        where={"source": fichier_pdf.name}
    )

    if anciens["ids"]:
        collection.delete(ids=anciens["ids"])

    # Indexer la nouvelle version
    collection.upsert(
        ids=ids,
        documents=textes,
        embeddings=embeddings,
        metadatas=metadonnees
    )

    return len(textes)


# ==========================================
# 4. SUPPRIMER UN DOCUMENT
# ==========================================

def supprimer_document(nom_fichier):
    """
    Supprime un document : ses chunks dans ChromaDB
    ET le fichier PDF du dossier data/pdf.
    Retourne le nombre de chunks supprimés.
    """

    # Vérifier que le nom est valide (pas de chemin)
    if Path(nom_fichier).name != nom_fichier:
        raise ValueError("Nom de fichier invalide.")

    # 1) Supprimer les chunks de ChromaDB
    resultats = collection.get(
        where={"source": nom_fichier}
    )

    nombre_chunks = len(resultats["ids"])

    if nombre_chunks:
        collection.delete(ids=resultats["ids"])

    # 2) Supprimer le fichier PDF
    chemin_pdf = DOSSIER_PDF / nom_fichier

    if chemin_pdf.exists():
        chemin_pdf.unlink()

    return nombre_chunks