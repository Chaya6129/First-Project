"""
כל החלטות התכנון של המערכת – במקום אחד.

חלק א' (הכנת הידע): שינוי של אחת ההגדרות כאן גורם לבנייה מחדש של המאגר.
המערכת מזהה את השינוי לבד ובונה מחדש – אין צורך למחוק את התיקייה ידנית.

חלק ב' (מענה לשאלה): אפשר לשנות בחופשיות, והשינוי נכנס לתוקף בהפעלה הבאה
בלי לגעת במאגר.
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"          # כאן שמים את קובץ הספר (txt / pdf / docx)
CHROMA_DIR = BASE_DIR / "chroma_db"   # כאן נשמר המאגר לדיסק
COLLECTION_NAME = "childrens_book"

# ───────────── חלק א': הכנת הידע (יקר לשינוי) ─────────────
CHUNK_SIZE = 300          # גודל קטע בטוקנים
CHUNK_OVERLAP = 30        # חפיפה בין קטעים (10%)
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 1536

# ───────────── חלק ב': מענה לשאלה (זול לשינוי) ─────────────
USE_QUERY_EXPANSION = True
# שמות המודלים: יש לוודא ב-Google AI Studio את השם המדויק של המודלים הזמינים לך
EXPANSION_MODEL = "gemini-3.1-flash-lite"   # Flash-Lite: מהיר וחסכוני להרחבת השאלה
ANSWER_MODEL = "gemini-3-flash-preview"     # Flash: ניסוח טוב יותר לתשובה
TOP_K = 3                                   # כמה קטעים לשלוף בכל שאלה
HISTORY_TURNS_FOR_EXPANSION = 2             # כמה סבבי שיחה קודמים לצרף להרחבה
