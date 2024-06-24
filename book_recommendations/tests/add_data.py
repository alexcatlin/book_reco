import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
pinecone_api_key=os.environ["PINECONE_API_KEY"]

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings

def add_document_to_vectorstore(index_name: str, file_type: str, file_path: str):
  # load text or pdf
  # args: name, filename, text or pdf
  """
  if text:
    loader = TextLoader
  if pdf:
    loader = PyPDFLoader
  if docx:
    loader = Docx2txtLoader
  if csv:
    loader = CSVLoader
  """


  pc = Pinecone(api_key=pinecone_api_key)
  if index_name not in pc.list_indexes().names():
    print("ERROR: Index does not exist. Please create the index first.")
    return None


  print(f"Loading {file_type} file: {file_path}...")
  loader = None
  # if (file_type == 'text'):
  #   loader = TextLoader(file_path)
  # elif(file_type == 'pdf'):
  #   loader = PyPDFLoader(file_path)
  # elif(file_type == 'docs'):
  #   loader = Docx2txtLoader(file_path)
  if file_type == 'csv':
    try:
        loader = CSVLoader(
            file_path=file_path,
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
                "fieldnames": ["Title",	"description",	"authors",	"image",	"previewLink",	"publisher",	"publishedDate",	"infoLink",	"categories", "ratingsCount"],
            },
            encoding="utf-8",
        )
        documents = loader.load()
        print("Documents loaded successfully:", len(documents))
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")

  documents = loader.load()
  print("Document length:", len(documents))

  # for each file content, decode and store filename as metadata
  filename = file_path.split("\\")[-1].replace(".docx", "")
  print("filename", filename)
  for document in documents:
    document.metadata['filename'] = filename

  # Initialize the RecursiveCharacterTextSplitter for splitting text
  # predefined length -- how many chars do we want per chunk
  # overlap - character 0 - 1000, first document. Then, there's an overlap of +-150 characters between doc 1 and doc 2
  print("Splitting text...")
  text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=150)
  chunked_documents = text_splitter.split_documents(documents)
  print("Chunk sample", chunked_documents[0])

  print('Length of chunks:', len(chunked_documents))
  
  print("Setting up embeddings and vectorstore...")
  embeddings = OpenAIEmbeddings()

  #todo: once jeremy gives documents for specific traits, add to namespace. but for now, no namespace
  pinecone = PineconeVectorStore.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    index_name=index_name,
    namespace=filename
  )
  
  # only used for actual text not from a loaded file
  # pinecone = PineconeVectorStore.from_texts(
  #   documents=chunked_documents,
  #   embedding=embeddings,
  #   index_name=index_name,
  #   namespace=""
  # )

  print("Document successfully added to Pinecone...")
  return pinecone


# def setup_training_data(index_name:str, directory_path:str = "training_data"):
#   print(f"Setting up training data from {directory_path}")
#   for filename in os.listdir(directory_path):
#     if filename.endswith(".docx"):
#       file_path = os.path.join(directory_path, filename)
#       add_document_to_vectorstore(index_name, "docs", file_path)


index_name = "recommendation"
file_type = "csv"
directory_path = "test"
for filename in os.listdir(directory_path):
  if filename.endswith(".csv"):
    if filename == 'test.csv':
      file_path = os.path.join(directory_path, filename)
      add_document_to_vectorstore(index_name=index_name, file_type=file_type, file_path=file_path)
      break