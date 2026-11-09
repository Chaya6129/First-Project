"""ממשק Streamlit לבוט. הרצה: streamlit run app.py"""
import truststore
truststore.inject_into_ssl()

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import config  # noqa: E402
from rag_engine import BookBot  # noqa: E402

st.set_page_config(page_title="שואלים את הספר", page_icon="📖", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Varela+Round&display=swap');
    .stApp, .stApp * { font-family: 'Varela Round', 'Arial Hebrew', Arial, sans-serif; }
    .main .block-container, section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
    [data-testid="stChatMessageContent"] { direction: rtl; text-align: right; }
    textarea { direction: rtl; }
    h1 { font-size: 2.2rem; letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="מכין את הספר…")
def get_bot() -> BookBot:
    return BookBot()


def render_debug(debug: dict) -> None:
    with st.expander("מאחורי הקלעים"):
        st.markdown(f"**השאלה כפי שנשאלה:** {debug['original']}")
        st.markdown(f"**השאלה אחרי הרחבה:** {debug['expanded']}")
        for i, chunk in enumerate(debug["chunks"], 1):
            score = f"{chunk['score']:.3f}" if chunk["score"] is not None else "?"
            st.markdown(f"**קטע {i}** · קרבה {score}")
            st.caption(chunk["text"])


st.title("📖 שואלים את הספר")
st.write("כתבו שאלה על הסיפור, והבוט יענה מתוך הספר.")

try:
    bot = get_bot()
except Exception as e:  # noqa: BLE001
    st.error(f"לא הצלחתי להכין את הבוט: {e}")
    st.stop()

with st.sidebar:
    st.header("מצב המערכת")
    st.write(f"קטעים במאגר: **{bot.chunk_count}**")
    st.write("המאגר נבנה עכשיו" if bot.built_now else "המאגר נטען מהדיסק")
    st.divider()
    st.caption(
        f"קטע: {config.CHUNK_SIZE} טוקנים · חפיפה: {config.CHUNK_OVERLAP}\n\n"
        f"Embedding: {config.EMBED_MODEL} ({config.EMBED_DIMENSIONS})\n\n"
        f"הרחבה: {config.EXPANSION_MODEL}\n\n"
        f"תשובה: {config.ANSWER_MODEL} · Top K: {config.TOP_K}"
    )
    show_debug = st.toggle("הצג מאחורי הקלעים", value=True)
    if st.button("התחל שיחה חדשה"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_debug and msg.get("debug"):
            render_debug(msg["debug"])

question = st.chat_input("מה תרצו לשאול על הספר?")
if question:
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("מחפש בספר…"):
                result = bot.ask(question, history)
        except Exception as e:  # noqa: BLE001
            text = str(e)
            if "429" in text or "quota" in text.lower() or "RESOURCE_EXHAUSTED" in text:
                st.error("הגעת למגבלת הבקשות של המסלול החינמי. חכו דקה ונסו שוב.")
            else:
                st.error(f"משהו השתבש: {e}")
            st.stop()

        st.markdown(result.answer)
        debug = {
            "original": result.original_question,
            "expanded": result.expanded_question,
            "chunks": [
                {"text": c.text, "score": c.score, "source": c.source}
                for c in result.chunks
            ],
        }
        if show_debug:
            render_debug(debug)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "debug": debug}
    )
