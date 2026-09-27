import os
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

# Настройка страницы
st.set_page_config(
    page_title="DocAI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocAI Assistant")
st.caption("Анализ документов с помощью искусственного интеллекта")
st.divider()

# Получение API ключа
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key не найден. Пожалуйста, укажите GOOGLE_API_KEY в Secrets.")
    st.stop()

# Настройка клиента Google AI
genai.configure(api_key=api_key)

# Загрузка PDF файла
uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file is not None:
    try:
        # Извлекаем текст напрямую из PDF
        reader = PdfReader(uploaded_file)
        document_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"

        if not document_text.strip():
            st.warning("Не удалось извлечь текст из PDF (возможно, это сканированный документ-картинка).")
            st.stop()

        st.success("Документ успешно загружен и прочитан!")

        st.divider()
        user_query = st.text_input("Задайте вопрос по документу:")

        if user_query:
            with st.spinner("Анализирую документ..."):
                prompt = f"""Вы — ассистент по анализу документов. Ответь на вопрос пользователя, используя только следующий текст документа.

Текст документа:
{document_text}

Вопрос: {user_query}
Ответ:"""

                # Инициализируем рабочую модель
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)

                st.write("### Ответ:")
                st.write(response.text)

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")