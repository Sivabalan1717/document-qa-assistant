from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from rag_pipeline import extract_text, split_text


# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_faiss_index(file_path):
    # 1. Extract text from the document
    text = extract_text(file_path)

    # 2. Split text into chunks
    chunks = split_text(text)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"Total chunks: {len(chunks)}")

    # 3. Convert chunks into numerical vectors
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print("Embedding shape:", embeddings.shape)

    # 4. Create the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Add the vectors to FAISS
    index.add(embeddings)

    print("FAISS index created successfully!")
    print("Vectors stored:", index.ntotal)

    return index, chunks, model


if __name__ == "__main__":
    file_path = input("Enter your PDF or TXT file path: ").strip()
    file_path = file_path.strip('"')

    try:
        index, chunks, model = build_faiss_index(file_path)
        print("\nIndex building completed!")

    except Exception as error:
        print("\nError:", error)
