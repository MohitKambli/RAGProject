from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

folder_path = "chroma_db"
cached_llm = Ollama(model="llama3.2")
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
	""" 
	<s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
	[INST] {input}
		Context: {context}
		Answer:
	[/INST]
	"""
)

def ask_pdf_post():
	query = 'Who is Alice?'
	print(f'Query: ', query)
	vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
	retriever = vector_store.as_retriever(
	search_type="similarity_score_threshold",
		search_kwargs={
			"k": 2,
			"score_threshold": 0.1,
		},
	)
	document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
	chain = create_retrieval_chain(retriever, document_chain)
	result = chain.invoke({"input": query})
	print('Chain Result: ', result)
	sources = []
	for doc in result["context"]:
		sources.append(
			{"source": doc.metadata["source"], "page_content": doc.page_content}
		)
	response_answer = {"answer": result["answer"], "sources": sources}
	print('Response Answer: ', response_answer)


def pdf_post():
	file_name = 'alice.pdf'
	save_file = "pdf/" + file_name
	loader = PDFPlumberLoader(save_file)
	docs = loader.load_and_split()
	print(f"docs len={len(docs)}")

	chunks = text_splitter.split_documents(docs)
	print(f"chunks len={len(chunks)}")

	vector_store = Chroma.from_documents(
		documents=chunks, embedding=embedding, persist_directory=folder_path
	)
	vector_store.persist()

	response = {
		"status": "Successfully Uploaded",
		"filename": file_name,
		"doc_len": len(docs),
		"chunks": len(chunks),
	}

	print('Response: ', response)
	ask_pdf_post()

pdf_post()