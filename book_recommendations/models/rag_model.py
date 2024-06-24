import faiss
import numpy as np
from datasets import Dataset, Features, Value, load_from_disk
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer

import config


def initializeRAG(merged_books_df):
    documents = (
        merged_books_df["Title"]
        + ", "
        + merged_books_df["description"]
        + ", "
        + merged_books_df["review/text"]
    ).tolist()

    # Create a Hugging Face Dataset
    features = Features({"id": Value("string"), "text": Value("string")})
    dataset = Dataset.from_dict(
        {"id": [str(i) for i in range(len(documents))], "text": documents},
        features=features,
    )

    # Create FAISS index
    d = 768  # Dimension of the embeddings (change this to match your actual embedding size)
    index = faiss.IndexFlatL2(d)

    # Generate or load embeddings (for example purposes, generating random embeddings)
    embeddings = np.random.random((len(documents), d)).astype(np.float32)

    # Add embeddings to the FAISS index
    index.add(embeddings)

    # Save the dataset and index
    dataset_path = config.DATASET_PATH
    index_path = config.INDEX_PATH

    dataset.save_to_disk(dataset_path)
    dataset.add_faiss_index("embeddings", custom_index=index)
    dataset.get_index("embeddings").save(index_path)

    print("Dataset and FAISS index saved successfully.")


def setup_rag_model():
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    dataset_path = config.DATASET_PATH
    index_path = config.INDEX_PATH
    dataset = load_from_disk(dataset_path)
    dataset.load_faiss_index("embeddings", index_path)
    rag_retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq",
        index_name="custom",
        passages_path=dataset_path,
        index_path=index_path,
        use_dummy_dataset=False,
    )
    rag_model = RagSequenceForGeneration.from_pretrained(
        "facebook/rag-sequence-nq", retriever=rag_retriever
    )
    return rag_tokenizer, rag_model


def rag_recommendation(query, rag_tokenizer, rag_model):
    inputs = rag_tokenizer(query, return_tensors="pt")
    generated_ids = rag_model.generate(**inputs)
    generated_text = rag_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text[0]
