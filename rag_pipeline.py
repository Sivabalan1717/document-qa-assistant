from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


MAX_PDF_PAGES = 40


def extract_text(file_path):
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(str(path))

        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(
                f"PDF has {len(reader.pages)} pages. "
                f"Maximum allowed is {MAX_PDF_PAGES} pages."
            )

        pages_text = []

        for page in reader.pages:
            pages_text.append(page.extract_text() or "")

        text = "\n".join(pages_text)

    elif extension == ".txt":
        text = path.read_text(encoding="utf-8-sig")

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not text.strip():
        raise ValueError(
            "No readable text found. The PDF may be scanned."
        )

    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_text(text)


if __name__ == "__main__":
    file_path = input("Enter the PDF or TXT file path: ").strip()
    file_path = file_path.strip('"')

    try:
        text = extract_text(file_path)
        chunks = split_text(text)

        print("\nText extraction successful!")
        print("Total characters:", len(text))
        print("Total chunks:", len(chunks))
        print("\nFirst chunk preview:\n")
        print(chunks[0][:500])

    except (ValueError, OSError) as error:
        print("\nError:", error)
