from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "nlpaueb/legal-bert-base-uncased"


model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


def get_answer(query, context):
    result = qa_pipeline(question=query, context=context)
    return result['answer']