"""
מנוע ה-RAG. כל תחנה בדרך מודפסת לטרמינל, כדי שאפשר יהיה לאבחן תקלות:
  קטע לא נשלף  ← הבעיה לפני החיפוש (חלוקה / Embedding / Top K / ניסוח השאלה)
  קטע נשלף אבל התשובה שגויה ← הבעיה אחרי החיפוש (פרומפט / מודל)
"""
import json
import os
from dataclasses import dataclass, field

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

EXPANSION_PROMPT = """אתה עוזר לחפש מידע בספר ילדים.
ילד שאל שאלה. לפעמים השאלה קצרה, לא ברורה, או מתייחסת למשהו שנאמר קודם בשיחה.
נסח אותה מחדש כשאלה אחת מלאה וברורה, באותה שפה, שתעזור למצוא את הקטע המתאים בספר.

כללים:
- שמור בדיוק על כוונת השאלה. אל תוסיף נושאים חדשים.
- אם השאלה מתייחסת לשיחה הקודמת (למשל "הוא", "היא", "ומה אחר כך?") – השלם את החסר מתוך השיחה.
- אם השאלה כבר ברורה – החזר אותה כמעט כמו שהיא.
- החזר רק את השאלה המנוסחת, בלי הסברים.

השיחה עד עכשיו:
{history}

שאלת הילד: {question}
שאלה מנוסחת:"""

ANSWER_PROMPT = """אתה מספר סיפורים חביב שעונה לילדים על שאלות על ספר.

כללים:
- ענה רק על סמך הקטעים מהספר שמופיעים למטה.
- אם התשובה לא נמצאת בקטעים, אמור בפשטות שזה לא כתוב בספר. אל תמציא.
- ענה בשפה פשוטה, חמה וקצרה שמתאימה לילדים, באותה שפה שבה נשאלה השאלה.

קטעים מהספר:
{context}

השאלה כפי שהילד שאל: {question}
כוונת השאלה (בניסוח מלא): {expanded}

תשובה:"""


@dataclass
class RetrievedChunk:
    text: str
    score: float | None
    source: str


@dataclass
class RagResult:
    answer: str
    original_question: str
    expanded_question: str
    chunks: list[RetrievedChunk] = field(default_factory=list)


def _part_a_signature() -> str:
    """'טביעת אצבע' של הגדרות חלק א'. אם היא משתנה – בונים את המאגר מחדש."""
    return json.dumps(
        {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "embed_model": config.EMBED_MODEL,
            "embed_dimensions": config.EMBED_DIMENSIONS,
        },
        sort_keys=True,
    )


class BookBot:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("לא נמצא GOOGLE_API_KEY. יש להגדיר אותו בקובץ .env")

        self.embed_model = GoogleGenAIEmbedding(
            model_name=config.EMBED_MODEL,
            api_key=api_key,
            embedding_config={"output_dimensionality": config.EMBED_DIMENSIONS},
        )
        self.expansion_llm = GoogleGenAI(
            model=config.EXPANSION_MODEL, api_key=api_key, temperature=0.2
        )
        self.answer_llm = GoogleGenAI(
            model=config.ANSWER_MODEL, api_key=api_key, temperature=0.3
        )

        self.index, self.chunk_count, self.built_now = self._load_or_build_index()
        self.retriever = self.index.as_retriever(similarity_top_k=config.TOP_K)

    # ───────────── חלק א': טעינה או בנייה של המאגר ─────────────
    def _load_or_build_index(self) -> tuple[VectorStoreIndex, int, bool]:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        signature = _part_a_signature()

        try:
            existing = client.get_collection(config.COLLECTION_NAME)
        except Exception:
            existing = None

        if existing is not None and existing.count() > 0:
            saved_signature = (existing.metadata or {}).get("settings")
            if saved_signature == signature:
                print(f"[מאגר] נמצא מאגר שמור עם {existing.count()} קטעים – טוען בלי Embedding.")
                vector_store = ChromaVectorStore(chroma_collection=existing)
                index = VectorStoreIndex.from_vector_store(
                    vector_store, embed_model=self.embed_model
                )
                return index, existing.count(), False
            print("[מאגר] הגדרות חלק א' השתנו – בונה את המאגר מחדש.")

        if existing is not None:
            client.delete_collection(config.COLLECTION_NAME)

        files = [p for p in config.DATA_DIR.glob("*") if p.is_file()] if config.DATA_DIR.exists() else []
        if not files:
            raise RuntimeError(f"לא נמצא קובץ ספר בתיקייה: {config.DATA_DIR}")

        print("[טעינה] קורא את הספר...")
        documents = SimpleDirectoryReader(str(config.DATA_DIR)).load_data()

        print(f"[חלוקה] מחלק לקטעים של {config.CHUNK_SIZE} טוקנים, חפיפה {config.CHUNK_OVERLAP}...")
        splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"[חלוקה] נוצרו {len(nodes)} קטעים.")

        collection = client.create_collection(
            config.COLLECTION_NAME,
            metadata={"settings": signature, "hnsw:space": "cosine"},
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("[Embedding] יוצר וקטורים ושומר במאגר...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )
        print("[מאגר] נשמר לדיסק. בהפעלה הבאה ייטען בלי Embedding.")
        return index, len(nodes), True

    # ───────────── חלק ב': מענה לשאלה ─────────────
    def _expand_question(self, question: str, history: list[dict]) -> str:
        if not config.USE_QUERY_EXPANSION:
            return question

        recent = history[-config.HISTORY_TURNS_FOR_EXPANSION * 2 :]
        history_text = "\n".join(
            f"{'ילד' if m['role'] == 'user' else 'בוט'}: {m['content']}" for m in recent
        ) or "(אין שיחה קודמת)"

        response = self.expansion_llm.complete(
            EXPANSION_PROMPT.format(history=history_text, question=question)
        )
        expanded = response.text.strip()
        return expanded or question

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        results = self.retriever.retrieve(query)
        return [
            RetrievedChunk(
                text=r.node.get_content(),
                score=r.score,
                source=r.node.metadata.get("file_name", ""),
            )
            for r in results
        ]

    def ask(self, question: str, history: list[dict]) -> RagResult:
        expanded = self._expand_question(question, history)
        print(f"\n[שאלה] מקורית: {question}")
        print(f"[שאלה] מורחבת: {expanded}")

        chunks = self._retrieve(expanded)
        for i, c in enumerate(chunks, 1):
            score = f"{c.score:.3f}" if c.score is not None else "?"
            print(f"[חיפוש] קטע {i} (קרבה {score}): {c.text[:80]}...")

        context = "\n\n---\n\n".join(c.text for c in chunks)
        response = self.answer_llm.complete(
            ANSWER_PROMPT.format(context=context, question=question, expanded=expanded)
        )
        answer = response.text.strip()
        print(f"[תשובה] {answer[:120]}")

        return RagResult(
            answer=answer,
            original_question=question,
            expanded_question=expanded,
            chunks=chunks,
        )
