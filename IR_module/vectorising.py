from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


document = '../uploads/testing.pdf'

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_embeddings_and_index(chunks, embedding_model):
    documents = [Document(page_content=str(chunk)) for chunk in chunks]
    db = FAISS.from_documents(documents, embedding_model)
    return db


# this function is going to carry out the search and combine the different documents for context
# using the langchain documentation, and their RAG tutorial
# some changes for improvement that could be made here are multiqueryretriever
def search_and_combine(query, db):
    context = ''
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
                               )
    retrieved_docs = retriever.invoke(query)
    for each in range(min(5, len(retrieved_docs))):
        context += retrieved_docs[each].page_content
        context += "\n\n"
    return context


