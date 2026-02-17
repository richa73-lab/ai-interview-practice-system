import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_file):
    """
    Input: pdf_file (uploaded file or file path)
    Output: extracted text as string
    """

    text = ""

    # If pdf_file is a Streamlit upload (bytes)
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except:
        # If pdf_file is a file path
        doc = fitz.open(pdf_file)

    for page in doc:
        text += page.get_text()

    return text
