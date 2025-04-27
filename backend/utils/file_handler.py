from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document

def handle_file_upload(file):
    """Handles file uploads and extracts text."""
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'docx':
        return extract_text_from_docx(file)
    elif file_extension == 'txt':
        return file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type")

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = Document(BytesIO(file.read()))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text
