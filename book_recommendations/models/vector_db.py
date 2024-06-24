import faiss
import numpy as np
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Function to generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


# Function to create FAISS index
def create_faiss_index(merged_books_df):
    embeddings = np.vstack(
        merged_books_df["description"].apply(generate_embeddings).values
    )
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # Build the index
    index.add(embeddings)  # Add embeddings to the index
    return index


# Function to query FAISS index
def query_faiss_index(description, index, merged_books_df, n=5):
    query_embedding = generate_embeddings(description)
    distances, indices = index.search(query_embedding, n)
    return merged_books_df.iloc[indices[0]]
