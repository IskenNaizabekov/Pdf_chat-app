import os
import streamlit as st
from pypdf import PdfReader
from google import genai

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

# Инициализация официального клиента Google GenAI
client = genai.Client(api_key=api_key)

# Загрузка PDF файла
uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file is not None:
    try:
        # Извлекаем весь текст напрямую из PDF
        reader = PdfReader(uploaded_file)
        document_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"

        st.success("Документ успешно загружен и прочитан!")

        st.divider()
        user_query = st.text_input("Задайте вопрос по документу:")

        if user_query:
            with st.spinner("Анализирую документ..."):
                # Формируем простой промпт с текстом документа
                prompt = f"""Вы — ассистент по анализу документов. Ответь на вопрос пользователя, используя только следующий текст документа.

Текст документа:
{document_text}

Вопрос: {user_query}
Ответ:"""

                # Вызываем генерацию через доступную модель gemini-2.0-flash
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )

                st.write("### Ответ:")
                st.write(response.text)

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")