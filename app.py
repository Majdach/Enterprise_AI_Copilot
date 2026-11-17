import streamlit as st
from pathlib import Path

from src.rag import poser_question_entreprise, collection
from src.ingestion import (
    indexer_pdf,
    supprimer_document,
    DOSSIER_PDF
)


# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================

st.set_page_config(
    page_title="Enterprise AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. PERSONNALISATION DU DESIGN
# ==========================================


st.markdown("""
<style>

/* Contenu principal */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Titres */
h1 {
    color: #60A5FA;
    font-weight: 800;
    letter-spacing: -0.8px;
}

h2, h3 {
    font-weight: 650;
}

/* Barre latérale */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 163, 184, 0.2);
}

/* Messages du chatbot */
[data-testid="stChatMessage"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Boutons */
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}

/* Champs de sélection */
[data-testid="stSelectbox"] {
    border-radius: 10px;
}

/* Statistiques */
[data-testid="stMetric"] {
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 12px;
}

/* Extraits documentaires */
[data-testid="stExpander"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Alertes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Séparateurs */
hr {
    border-color: rgba(148, 163, 184, 0.15);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. INITIALISER L'ÉTAT DE LA SESSION
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sert à vider le champ d'import après une indexation ou une suppression
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ==========================================
# 4. BARRE LATÉRALE
# ==========================================

with st.sidebar:

    st.title("📚 Documents")

    # Message affiché après une action (indexation / suppression)
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    # Récupérer les documents indexés
    donnees = collection.get(
        include=["metadatas"]
    )

    sources = sorted({
        metadata["source"]
        for metadata in donnees["metadatas"]
        if metadata and "source" in metadata
    })

    # ======================================
    # SÉLECTION DU DOCUMENT
    # ======================================

    st.subheader("🔎 Recherche documentaire")

    document_selectionne = st.selectbox(
        "Choisissez un document :",
        ["Tous les documents"] + sources
    )

    source_selectionnee = (
        None
        if document_selectionne == "Tous les documents"
        else document_selectionne
    )

    st.divider()

    # ======================================
    # IMPORTATION DES PDF
    # ======================================

    st.subheader("📤 Importer un document")

    fichier_importe = st.file_uploader(
        "Choisissez un fichier PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if fichier_importe is not None:

        if st.button(
            "Indexer le document",
            key="indexer_document"
        ):

            with st.spinner("Indexation en cours..."):

                try:
                    # Nettoyer le nom du fichier
                    nom_fichier = Path(
                        fichier_importe.name
                    ).name

                    chemin = DOSSIER_PDF / nom_fichier

                    # Sauvegarder le PDF
                    contenu = fichier_importe.getvalue()
                    chemin.write_bytes(contenu)

                    # Indexer dans ChromaDB
                    nombre_chunks = indexer_pdf(chemin)

                    st.session_state.flash = (
                        f"Document indexé : {nombre_chunks} chunks."
                    )

                    # Vider le champ d'import et actualiser
                    st.session_state.uploader_key += 1
                    st.rerun()

                except Exception as erreur:

                    st.error(
                        f"Erreur pendant l'indexation : {erreur}"
                    )

    st.divider()

    # ======================================
    # DOCUMENTS DISPONIBLES
    # ======================================

    st.subheader("📄 Documents disponibles")

    if sources:
        for source in sources:
            st.write("📄", source)
    else:
        st.info("Aucun document indexé.")

    st.divider()

    # ======================================
    # SUPPRIMER UN DOCUMENT
    # ======================================

    st.subheader("🗑️ Supprimer un document")

    if sources:

        document_a_supprimer = st.selectbox(
            "Document à supprimer :",
            sources,
            key="document_a_supprimer"
        )

        if st.button(
            "Supprimer définitivement",
            key="supprimer_document"
        ):

            try:
                nombre_supprimes = supprimer_document(
                    document_a_supprimer
                )

                st.session_state.flash = (
                    f"Document supprimé "
                    f"({nombre_supprimes} chunks retirés)."
                )

                # Vider aussi le champ d'import pour éviter une réindexation
                st.session_state.uploader_key += 1
                st.rerun()

            except Exception as erreur:

                st.error(
                    f"Erreur pendant la suppression : {erreur}"
                )

    else:
        st.info("Aucun document à supprimer.")

    st.divider()

    # ======================================
    # STATISTIQUES
    # ======================================

    st.subheader("📊 Statistiques")

    st.metric(
        "Documents",
        len(sources)
    )

    st.metric(
        "Chunks indexés",
        collection.count()
    )

    st.divider()

    # ======================================
    # CONFIGURATION
    # ======================================

    st.subheader("⚙️ Configuration")

    st.write("**LLM :** Qwen 2.5:3b")
    st.write("**Embeddings :** MiniLM")
    st.write("**Base vectorielle :** ChromaDB")

    st.divider()

    # ======================================
    # EFFACER LA CONVERSATION
    # ======================================

    if st.button("🗑️ Effacer la conversation"):

        st.session_state.messages = []
        st.rerun()


# ==========================================
# 5. TITRE PRINCIPAL
# ==========================================

st.title("Enterprise AI Copilot")

st.markdown("#### Your Intelligent Document Assistant")

st.caption(
    "Assistant IA local propulsé par Qwen 2.5, "
    "Sentence Transformers et ChromaDB."
)

st.divider()


# ==========================================
# 6. DOCUMENT ACTUEL
# ==========================================

if source_selectionnee is None:

    st.info(
        "🔎 Recherche active : Tous les documents"
    )

else:

    st.info(
        f"🔎 Recherche active : {source_selectionnee}"
    )


# ==========================================
# 7. MESSAGE DE BIENVENUE
# ==========================================

if not st.session_state.messages:

    st.info(
        "👋 Bienvenue ! Posez une question "
        "sur vos documents d'entreprise."
    )


# ==========================================
# 8. AFFICHER L'HISTORIQUE
# ==========================================


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        sources_message = message.get("sources", [])

        if message["role"] == "assistant" and sources_message:

            with st.expander("📄 Voir l'extrait justificatif"):

                for source in sources_message:

                    st.caption(
                        f"📄 {source['fichier']} "
                        f"— Page {source['page']}"
                    )

                    st.info(source["extrait"])

# ==========================================
# 9. ZONE DE DISCUSSION
# ==========================================

question = st.chat_input(
    "Posez une question sur vos documents..."
)


# ==========================================
# 10. RÉPONSE DU RAG
# ==========================================

if question:

    # Enregistrer la question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Afficher la question
    with st.chat_message("user"):
        st.markdown(question)

    # Générer la réponse
    with st.chat_message("assistant"):

        with st.spinner("Qwen analyse vos documents..."):

            try:

                resultat = poser_question_entreprise(
                    question,
                    source=source_selectionnee
                )

                reponse = resultat["reponse"]
                sources_retrouvees = resultat["sources"]

            except Exception as erreur:

                st.error(
                    "Erreur lors de la génération. "
                    "Vérifiez Ollama et ChromaDB."
                )

                st.exception(erreur)
                st.stop()

        st.markdown(reponse)

        # Afficher les passages récupérés
        with st.expander("📚 Voir les passages retrouvés"):

            if sources_retrouvees:

                for i, source in enumerate(
                    sources_retrouvees, start=1
                ):
                    st.markdown(
                        f"**Passage {i}**"
                    )

                    st.write(
                        "📄 Fichier :",
                        source["fichier"]
                    )

                    st.write(
                        "📍 Page :",
                        source["page"]
                    )

                    st.write(
                        "📝 Extrait :"
                    )

                    st.info(source["extrait"])

                    st.divider()

            else:
                st.info("Aucun passage retrouvé.")


    # Enregistrer la réponse
    st.session_state.messages.append({
        "role": "assistant",
        "content": reponse
    })