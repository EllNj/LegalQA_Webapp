from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os


def text_from_pdf(pdf_path):# im thinking that to implement this in a web app, maybe generate ID
    text = ''
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for num in range(len(reader.pages)):
            text += reader.pages[num].extract_text()
    return text

def create_chunks(text, chunk_size=512, overlap=60):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks




def text_to_chunks(pdf_path):
    return create_chunks(text_from_pdf(pdf_path))
