import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="DocAI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocAI Assistant")
st.caption("Анализ документов с помощью искусственного интеллекта")
st.divider()

api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key is missing.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

uploaded_file = st.file_uploader("Загрузите PDF документ", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.success("Документ загружен!")

    @st.cache_resource(show_spinner="Обработка документа...")
    def process_pdf(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore

    try:
        vectorstore = process_pdf(tmp_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        system_prompt = (
            "Ты — умный ассистент по анализу документов.\n"
            "Тебе предоставлены фрагменты документа.\n"
            "Пользователь задаёт вопрос на русском языке.\n"
            "Твоя задача:\n"
            "1. Ответить на вопрос подробно и понятно НА РУССКОМ ЯЗЫКЕ.\n"
            "2. В конце ответа ОБЯЗАТЕЛЬНО укажи номера страниц, откуда взята информация (например: 'Источник: Страница 12').\n"
            "3. Если информации нет в тексте, честно ответь, что в документе этого нет.\n\n"
            "Контекст из документа:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        user_query = st.text_input("Задайте вопрос по документу:")

        if user_query:
            with st.spinner("Поиск ответа..."):
                response = rag_chain.invoke({"input": user_query})
                
                st.markdown("### Ответ:")
                st.info(response["answer"])

                with st.expander("Источники"):
                    for doc in response["context"]:
                        page_num = doc.metadata.get("page", 0) + 1
                        st.write(f"📌 *Страница {page_num}:*")
                        st.caption(doc.page_content[:300] + "...")

    except Exception as e:
        st.error(f"Ошибка: {e}")