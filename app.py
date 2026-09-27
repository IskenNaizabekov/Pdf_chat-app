import os
import tempfile
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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

os.environ["GOOGLE_API_KEY"] = api_key

# Загрузка PDF файла
uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()
        st.success("Документ успешно загружен!")

        # Инициализация модели Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        st.divider()
        user_query = st.text_input("Задайте вопрос по документу:")

        if user_query:
            with st.spinner("Анализирую документ..."):
                # Передаем PDF-файл напрямую в модель Gemini как медиа-данные
                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"Проанализируй документ и ответь на вопрос: {user_query}"
                        },
                        {
                            "type": "media",
                            "mime_type": "application/pdf",
                            "data": pdf_bytes
                        }
                    ]
                )
                
                response = llm.invoke([message])
                
                st.write("### Ответ:")
                st.write(response.content)

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")