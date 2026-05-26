import os


def extract_text_from_pdf(pdf_path):
    """Try to extract text from PDF using pdfplumber if available.

    If pdfplumber is not installed, function returns an empty string and
    prints a friendly message. This keeps the pipeline runnable in
    minimal environments.
    """

    text = ""

    try:
        import pdfplumber

    except Exception:
        print("pdfplumber not available; skipping PDF text extraction.")
        return text

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return text


def load_documents(folder_path):

    documents = {}

    for file in os.listdir(folder_path):

        path = os.path.join(folder_path, file)

        # PDF files
        if file.endswith(".pdf"):

            text = extract_text_from_pdf(path)

            documents[file] = text

        # TXT files
        elif file.endswith(".txt"):

            try:
                with open(path, "r", encoding="utf-8") as f:
                    documents[file] = f.read()

            except Exception as e:
                print(f"Error reading {file}: {e}")

    return documents