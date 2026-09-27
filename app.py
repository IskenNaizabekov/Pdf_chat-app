import os
import streamlit as st
from google import genai

st.set_page_config(
    page_title="PDF Нурбол",
    page_icon="📚"
)

st.title("📚 PDF Нурбол")
st.write("Загрузи учебник или другой PDF и задай вопрос.")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Не найден GOOGLE_API_KEY")
    st.stop()

client = genai.Client(api_key=api_key)

pdf = st.file_uploader(
    "📄 Загрузить PDF",
    type=["pdf"]
)

if pdf:
    st.success(f"Файл загружен: {pdf.name}")

    if "uploaded_file" not in st.session_state:
        with st.spinner("Анализирую PDF..."):
            temp_path = "uploaded_document.pdf"

            with open(temp_path, "wb") as f:
                f.write(pdf.getvalue())

            st.session_state.uploaded_file = client.files.upload(
                file=temp_path
            )

        st.success("PDF готов к вопросам!")

    question = st.text_input(
        "❓ Твой вопрос",
        placeholder="Например: Что говорится в документе о системе образования?"
    )

    if st.button("Получить ответ") and question:

        prompt = f"""
Ты помощник по учебным документам.

Отвечай на языке на котором был введен запрос .

Используй только информацию из загруженного PDF.

Вопрос:
{question}

Дай понятный и краткий ответ.
Если возможно, укажи страницу, где найдена информация.
Если ответа в документе нет, честно скажи об этом.
"""

        with st.spinner("Ищу ответ в документе..."):
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    prompt,
                    st.session_state.uploaded_file
                ]
            )

        st.subheader("💬 Ответ")
        st.write(response.text)