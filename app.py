import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Настройка страницы
st.set_page_config(
    page_title="DocAI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocAI Assistant")
st.caption("Анализ документов с помощью искусственного интеллекта")
st.divider()

# Получение API ключа из Secrets или переменных окружения
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key is missing. Пожалуйста, укажите GOOGLE_API_KEY в Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# Загрузка PDF файла
uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file is not None:
    try:
        # Сохранение во временный файл для чтения PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        st.success("Документ загружен!")

        # 1. Загрузка текста из PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Удаляем временный файл
        os.remove(tmp_path)

        # 2. Разделение текста на чанки
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Векторное хранилище в памяти (не требует сторонних C-библиотек)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = DocArrayInMemorySearch.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # 4. Настройка генеративной модели и Промпта
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        system_prompt = (
            "Вы — ассистент по анализу документов. Используйте следующий контекст, "
            "чтобы ответить на вопрос пользователя. Если ответа нет в контексте, "
            "честно скажите, что не знаете.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 5. Создание RAG цепи
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Чат-интерфейс
        st.divider()
        user_query = st.text_input("Задайте вопрос по документу:")

        if user_query:
            with st.spinner("Анализирую документ..."):
                response = rag_chain.invoke({"input": user_query})
                st.write("### Ответ:")
                st.write(response["answer"])

    except Exception as e:
        st.error(f"Ошибка: {e}")