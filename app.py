import os
import streamlit as st
from google import genai

st.set_page_config(
    page_title="PDF Учитель",
    page_icon="📚"
)

st.title("📚 PDF Учитель")
st.write("Загрузи PDF и задай вопрос по документу.")

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
            try:
                temp_path = "uploaded_document.pdf"

                with open(temp_path, "wb") as f:
                    f.write(pdf.getvalue())

                st.session_state.uploaded_file = client.files.upload(
                    file=temp_path
                )

            except Exception as e:
                st.error("Не удалось загрузить PDF.")
                st.code(str(e))
                st.stop()

        st.success("PDF готов к вопросам!")

    question = st.text_input(
        "❓ Твой вопрос",
        placeholder="Например: Что говорится в документе о системе образования?"
    )

    if st.button("Получить ответ") and question:

        prompt = f"""
Ты помощник по учебным документам.

Определи язык вопроса пользователя.

Отвечай на том же языке, на котором задан вопрос:
- русский вопрос → русский ответ
- кыргызский вопрос → кыргызский ответ
- английский вопрос → английский ответ

Используй только информацию из загруженного PDF.

Вопрос:
{question}

Дай понятный и краткий ответ.

Если возможно, укажи страницу, где найдена информация.

Если ответа в PDF нет, честно скажи об этом.
"""

        with st.spinner("Ищу ответ в документе..."):
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
                st.error("Не удалось получить ответ от Gemini.")
                st.code(str(e))