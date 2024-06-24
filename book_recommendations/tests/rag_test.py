from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY=os.environ["PINECONE_API_KEY"]
use_serverless=os.environ["USE_SERVERLESS"]

EMBEDDING_MODEL = "text-embedding-ada-002"

def setup_vectorstore(index_name):
  embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
  pinecone_vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

  return pinecone_vectorstore

vectorstore = setup_vectorstore("recommendation")
retriever = vectorstore.as_retriever()

# GPT_MODEL = "gpt-3.5-turbo-0125"
GPT_MODEL = "gpt-4o"
llm = ChatOpenAI(model=GPT_MODEL, temperature=0)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

prompt = """You are an assistant that recommends books. The user will give you some details of a book (title, genre, etc.). You need to recommend a book that is similar to what the user is looking for. The recommended book should be in the same genre and have a similar theme. The recommended book should also be popular and well-received by readers. You can assume that the user is looking for a book that is similar to the given book in terms of genre, theme, and popularity. You can also assume that the user is open to reading books from different authors and publishers. You can use any information available to you to make the recommendation. You can also ask the user for more information if needed.


User Prompt: {query}
"""

# rag_chain = (
#     ({"context": retriever | format_docs, "question": RunnablePassthrough()})
#     | prompt
#     | llm
#     | StrOutputParser()
# )

user_input = input("Enter book details:")
print("input", user_input)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt,
    },
)

print("here")

# output = rag_chain.invoke(user_input)
output = chain.invoke({"query": user_input})
print("Output:", output)