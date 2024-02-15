# Article-Based-Question-and-Answer-System-Using-LLM-PROJECT-REPORT
Develop an intelligent question-answering system that can accurately answer questions posed 
about a given article. The system should utilize large language models (LLMs) to effectively 
process and understand the context of both the article and the question, enabling it to generate 
2
comprehensive and relevant answers. The system should be able to handle a variety of question 
types, including open-ended, challenging, and strange questions. Additionally, the system 
should provide evidence from the article to support its answers, enhancing the transparency and 
credibility of its responses

![Screenshot 2024-02-15 132006](https://github.com/Ganesh517/Article-Based-Question-and-Answer-System-Using-LLM-PROJECT-REPORT/assets/75235006/25788bc2-601a-4552-a5ae-9ef9ebaff2d8)

## main.py
```python
import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
#load_dotenv()  # take environment variables from .env (especially openai api key)
os.environ['OPENAI_API_KEY'] = ''

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)



```

## OUTPUT

![Screenshot 2023-11-20 101005](https://github.com/Ganesh517/Article-Based-Question-and-Answer-System-Using-LLM-PROJECT-REPORT/assets/75235006/dae21022-b0e5-4a49-9bab-006d2b4e1e40)

