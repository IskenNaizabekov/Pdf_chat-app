import os
import time
import streamlit as st
from google import genai

st.set_page_config(
    page_title="PDF Учитель",
    page_icon="📚"
)

st.title("📚 PDF Учитель")
st.write("Загрузи PDF и задай вопрос по документу.")

# API ключ
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("⚠️ API ключ не настроен.")
    st.stop()

client = genai.Client(api_key=api_key)

# Загрузка PDF
pdf = st.file_uploader(
    "📄 Загрузить PDF",
    type=["pdf"]
)

if pdf:

    st.success(f"Файл загружен: {pdf.name}")

    # Загружаем PDF только один раз
    if (
        "uploaded_file" not in st.session_state
        or st.session_state.get("file_name") != pdf.name
    ):

        with st.spinner("📖 Анализирую PDF..."):

            try:
                temp_path = "uploaded_document.pdf"

                with open(temp_path, "wb") as f:
                    f.write(pdf.getvalue())

                uploaded_file = client.files.upload(
                    file=temp_path
                )

                st.session_state.uploaded_file = uploaded_file
                st.session_state.file_name = pdf.name

            except Exception as e:
                st.error("❌ Не удалось загрузить PDF.")
                st.stop()

        st.success("✅ PDF готов к вопросам!")

    # Вопрос
    question = st.text_input(
        "❓ Твой вопрос",
        placeholder="Например: Объясни содержание этого документа"
    )

    if st.button("Получить ответ") and question:

        prompt = f"""
Ты — помощник по учебным документам.

Пользователь задаёт вопрос по загруженному PDF.

Определи язык вопроса.

Если вопрос на русском — отвечай на русском.
Если вопрос на кыргызском — отвечай на кыргызском.
Если вопрос на английском — отвечай на английском.

Используй только информацию из загруженного PDF.

Не придумывай информацию, которой нет в документе.

Отвечай понятно и простым языком.

Если возможно, укажи страницу, где находится ответ.

Если ответа в документе нет, честно сообщи об этом.

Вопрос пользователя:
{question}
"""

        with st.spinner("🤖 Ищу ответ в документе..."):

            try:

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[
                        prompt,
                        st.session_state.uploaded_file
                    ]
                )

                st.subheader("💬 Ответ")
                st.write(response.text)

            except Exception as e:

                error_text = str(e)

                if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                    st.warning(
                        "⏳ Сейчас AI перегружен или достигнут лимит запросов. "
                        "Попробуйте повторить запрос через некоторое время."
                    )

                else:

                    st.error(
                        "❌ Не удалось получить ответ от AI."
                    )