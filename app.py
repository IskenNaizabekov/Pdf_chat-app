import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Настройка интерфейса Streamlit
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
    st.error("API Key не найден. Пожалуйста, укажите GOOGLE_API_KEY в Secrets.")
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

        st.success("Документ успешно загружен!")

        # 1. Загрузка текста из PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Удаление временного файла после чтения
        os.remove(tmp_path)

        # 2. Разделение текста на фрагменты (чанки)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Векторное хранилище в памяти без внешних зависимостей C/C++
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # 4. Настройка языковой модели Gemini и промпта
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

        # 5. Сборка RAG-цепи
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Интерактивный вопрос-ответ
        st.divider()
        user_query = st.text_input("Задайте вопрос по документу:")

        if user_query:
            with st.spinner("Анализирую документ..."):
                response = rag_chain.invoke({"input": user_query})
                st.write("### Ответ:")
                st.write(response["answer"])

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")