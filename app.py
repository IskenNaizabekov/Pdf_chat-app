import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Настройка страницы Streamlit
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
        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        st.success("Документ успешно загружен!")

        # 1. Загрузка текста из PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Удаление временного файла после чтения
        os.remove(tmp_path)

        # 2. Разделение текста на чанки
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Векторное хранилище в оперативной памяти
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # 4. Модель Gemini с явным вызовом v1beta
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3,
            client_options=None
        )

        template = """Вы — ассистент по анализу документов. Используйте следующий контекст, чтобы ответить на вопрос пользователя. Если ответа нет в контексте, честно скажите, что не знаете.

Контекст:
{context}

Вопрос: {question}
Ответ:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Функция форматирования текста из фрагментов
        def format_docs(docs_list):
            return "\n\n".join(doc.page_content for doc in docs_list)

        # 5. Сборка RAG-цепи через LCEL
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Интерфейс вопросов и ответов
        st.divider()
        user_query = st.text_input("Задайте вопрос по документу:")

        if user_query:
            with st.spinner("Анализирую документ..."):
                answer = rag_chain.invoke(user_query)
                st.write("### Ответ:")
                st.write(answer)

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")