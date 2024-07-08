from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import numpy as np
from preprocess_doc import text_to_chunks
from langchain.schema import Document

document = '../uploads/testing.pdf'
chunks = text_to_chunks(document)
documents = [Document(page_content=chunk)for chunk in chunks]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


db = FAISS.from_documents(documents, embedding_model)

print(db.index.ntotal)

query = "would my AI that sees if someone has good credit be considered high risk?"
docs = db.similarity_search(query)
print(f"this is chunk 1, {docs[0].page_content}")
print(f"this is chunk 2, {docs[1].page_content}")
print(f"this is chunk 3, {docs[2].page_content}")

# git testing 123