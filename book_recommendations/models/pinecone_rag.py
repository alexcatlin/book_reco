from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

import book_recommendations.config

EMBEDDING_MODEL = "text-embedding-ada-002"

def setup_vectorstore(index_name):
  embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
  pinecone_vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

  return pinecone_vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def query_pinecone(description):
  vectorstore = setup_vectorstore("recommendation")
  retriever = vectorstore.as_retriever()

  # GPT_MODEL = "gpt-3.5-turbo-0125"
  GPT_MODEL = "gpt-4o"
  llm = ChatOpenAI(model=GPT_MODEL, temperature=0)

  

  prompt = """You are an assistant that recommends books. The user will give you some details of a book (title, genre, etc.). You need to recommend a book that is similar to what the user is looking for. The recommended book should be in the same genre and have a similar theme. The recommended book should also be popular and well-received by readers. You can assume that the user is looking for a book that is similar to the given book in terms of genre, theme, and popularity. You can also assume that the user is open to reading books from different authors and publishers. You can use any information available to you to make the recommendation. You can also ask the user for more information if needed.


  User Prompt: {query}
  """

  #not sure why LCEL pipeline is working -- following docs
  # rag_chain = (
  #     ({"context": retriever | format_docs, "question": RunnablePassthrough()})
  #     | prompt
  #     | llm
  #     | StrOutputParser()
  # )


  #not sure why it's giving an error about a dictionary -- following docs
  chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt,
    },
  )

  output = chain.invoke({"query": description})
  print("Output:", output)

  return output