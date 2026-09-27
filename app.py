import streamlit as st
from google import genai
import tempfile
import os

st.set_page_config(
    page_title="PDF Помощник",
    page_icon="📘"
)

st.title("📘 PDF Помощник")
st.write("Загрузите PDF и задайте вопрос по документу.")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
except:
    st.error("Не найден API-ключ.")
    st.stop()

pdf = st.file_uploader(
    "Выберите PDF-файл",
    type="pdf"
)

if pdf:

    if "file_name" not in st.session_state or \
       st.session_state.file_name != pdf.name:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as file:

            file.write(pdf.getvalue())
            file_path = file.name

        try:
            st.session_state.pdf_file = client.files.upload(
                file=file_path
            )
            st.session_state.file_name = pdf.name

        finally:
            os.remove(file_path)

        st.success("PDF загружен.")

    question = st.text_input(
        "Ваш вопрос:"
    )

    if st.button("Получить ответ"):

        if question:

            prompt = f"""
Ответь на вопрос по содержанию загруженного PDF.

Используй только информацию из документа.
Не придумывай информацию.
Отвечай понятно и на русском языке.
Если ответа в документе нет, так и напиши.

Вопрос:
{question}
"""

            with st.spinner("Ищу ответ..."):

                try:
                    answer = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            prompt,
                            st.session_state.pdf_file
                        ]
                    )

                    st.subheader("Ответ")
                    st.write(answer.text)

                except Exception as error:
                    st.error(f"Ошибка: {error}")

        else:
            st.warning("Введите вопрос.")

else:
    st.info("Сначала загрузите PDF.")