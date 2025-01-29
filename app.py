from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os

# Folder path for vector DB
folder_path = "chroma_db"

# Initialize LLM and Embedding
cached_llm = Ollama(model="llama3")
embedding = FastEmbedEmbeddings()

# Text Splitter Configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Fixing Prompt Formatting
raw_prompt = PromptTemplate.from_template(
    """
    You are a technical assistant good at searching documents. 
    If you do not have an answer from the provided information, say so.

    Question: {input}
    Context: {context}

    Answer:
    """
)

def ask_pdf_post():
    # query = 'Who is Alice?'
    # query = 'What was Trysdale doing?'
    query = 'Who is Antonio?'
    print(f'Query: {query}')

    # Reload the persisted vector store
    if not os.path.exists(folder_path):
        print("Error: Vector database not found.")
        return
    
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.1},  # Reduce k for precision
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    # Invoke the retrieval and LLM chain
    result = chain.invoke({"input": query})

    response_answer = {"answer": result.get("answer", "No answer found.")}
    print('Response Answer:', response_answer)


def pdf_post():
    file_name = 'mov.pdf'
    save_file = "pdf/" + file_name

    # Load and split PDF
    loader = PDFPlumberLoader(save_file)
    docs = loader.load()
    
    # Ensure text is split using the text_splitter
    chunks = text_splitter.split_documents(docs)

    print(f"Total documents: {len(docs)}, Total chunks: {len(chunks)}")

    # Store embeddings in Chroma DB
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()

    print("PDF Processed and Stored in Vector Database.")
    
    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }

    print('Response:', response)
    
    # Ask the stored data after processing
    ask_pdf_post()

pdf_post()


'''
a, b = 3, 5
a = a ^ b
b = a ^ b
a = a ^ b
print(a, b)
'''