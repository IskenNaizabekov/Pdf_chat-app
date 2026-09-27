import os
import tempfile
import streamlit as st
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

# Инициализация API
genai.configure(api_key=api_key)

# Загрузка PDF файла
uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file is not None:
    # Сохраняем временный файл для отправки в Google File API
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        st.success("Документ успешно загружен!")
        st.divider()
        user_query = st.text_input("Задайте любой вопрос по документу:")

        if user_query:
            with st.spinner("ИИ обрабатывает документ..."):
                # Загружаем PDF напрямую на серверы Google
                google_file = genai.upload_file(tmp_path, mime_type="application/pdf")
                
                # Используем стандартную модель gemini-1.5-flash
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Отправляем файл и вопрос
                response = model.generate_content([google_file, user_query])

                st.write("### Ответ:")
                st.write(response.text)

                # Удаляем временный файл с серверов Google
                genai.delete_file(google_file.name)

    except Exception as e:
        st.error(f"Произошла ошибка при обработке: {e}")
    finally:
        # Удаляем локальный временный файл
        if os.path.exists(tmp_path):
            os.remove(tmp_path)