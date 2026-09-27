import os
import tempfile
import streamlit as st
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

# Инициализация нового клиента Google GenAI
client = genai.Client(api_key=api_key)

# Загрузка PDF файла
uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file is not None:
    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        st.success("Документ успешно загружен!")
        st.divider()
        user_query = st.text_input("Задайте любой вопрос по документу:")

        if user_query:
            with st.spinner("ИИ обрабатывает документ..."):
                # Загружаем файл через новый SDK
                google_file = client.files.upload(file=tmp_path)
                
                # Отправляем запрос
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=[google_file, user_query]
                )

                st.write("### Ответ:")
                st.write(response.text)

                # Удаляем файл с серверов
                client.files.delete(name=google_file.name)

    except Exception as e:
        st.error(f"Произошла ошибка при обработке: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)