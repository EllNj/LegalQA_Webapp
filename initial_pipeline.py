from IR_module import preprocess_doc, vectorising
from langchain_huggingface import HuggingFaceEmbeddings
from Training_QA_model import untrained_legalbert


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
document_path = "./uploads/testing.pdf"
chunks = preprocess_doc.text_to_chunks(document_path)

db = vectorising.create_embeddings_and_index(chunks=chunks,embedding_model=embedding_model)

query = "what does 'AI system' mean?"
context = vectorising.search_and_combine(query=query, db=db)

answer = untrained_legalbert.get_answer(query=query, context=context)

print(query)
print(answer)
