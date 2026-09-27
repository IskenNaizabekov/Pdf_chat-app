import streamlit as st
from pypdf import PdfReader
from g4f.client import Client

# Настройка страницы
st.set_page_config(
    page_title="DocAI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocAI Assistant")
st.caption("Анализ документов с помощью искусственного интеллекта")
st.divider()

# Инициализация бесплатного AI клиента
client = Client()

# Загрузка PDF файла
uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file is not None:
    try:
        # Извлекаем весь текст из PDF
        reader = PdfReader(uploaded_file)
        document_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"

        if not document_text.strip():
            st.warning("Не удалось извлечь текст из PDF (возможно, это скан или картинка).")
            st.stop()

        st.success("Документ успешно загружен и прочитан!")

        st.divider()
        user_query = st.text_input("Задайте любой вопрос по документу:")

        if user_query:
            with st.spinner("ИИ считывает и обрабатывает документ..."):
                prompt = f"""Ты — умный ассистент по анализу документов.
Ответь на вопрос пользователя максимально подробно и понятно, используя следующий текст документа.

Текст документа:
{document_text}

Вопрос пользователя: {user_query}
Ответ:"""

                # Отправка запроса в бесплатный рабочий провайдер GPT
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content

                st.write("### Ответ:")
                st.write(answer)

    except Exception as e:
        st.error(f"Произошла ошибка при обработке: {e}")