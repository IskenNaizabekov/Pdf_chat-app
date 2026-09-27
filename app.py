import streamlit as st
from google import genai
import tempfile
import os

st.set_page_config(
    page_title="PDF Помощник",
    page_icon="📘"
)

st.title("📘 PDF Помощник")
st.write("Загрузите PDF и задайте вопрос.")

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API-ключ не найден.")
    st.stop()

client = genai.Client(
    api_key=st.secrets["GOOGLE_API_KEY"]
)

pdf = st.file_uploader(
    "Выберите PDF-файл",
    type=["pdf"]
)

if pdf:

    if "pdf_file" not in st.session_state:

        with st.spinner("Загрузка PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as file:

                file.write(pdf.getvalue())
                path = file.name

            try:
                st.session_state.pdf_file = client.files.upload(
                    file=path
                )
            finally:
                os.remove(path)

        st.success("PDF загружен.")

    question = st.text_input("Ваш вопрос")

    if st.button("Получить ответ"):

        if not question:
            st.warning("Введите вопрос.")
            st.stop()

        prompt = f"""
Ответь на вопрос, используя загруженный PDF.

Используй только информацию из документа.
Не придумывай информацию.
Отвечай понятно на русском языке.
Если ответа нет в документе, скажи об этом.

Вопрос:
{question}
"""

        with st.spinner("Ищу ответ..."):

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        prompt,
                        st.session_state.pdf_file
                    ]
                )

                st.subheader("Ответ")
                st.write(response.text)

            except Exception as e:
                st.error("Ошибка при обработке PDF.")
                st.code(str(e))

else:
    st.info("Загрузите PDF-файл.")