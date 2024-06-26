import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY")
ASSISTANCE_API_KEY = os.getenv("ASSISTANCE_API_KEY")
BOOKS_FILE = os.getenv("BOOKS_FILE")
BOOK_RATINGS_FILE = os.getenv("BOOK_RATINGS_FILE")
MERGED_BOOKS_FILE = os.getenv("MERGED_BOOKS_FILE")
ROW_NUMBER_TO_TEST = int(os.getenv("ROW_NUMBER_TO_TEST"))
DATASET_PATH = os.getenv("DATASET_PATH")
INDEX_PATH = os.getenv("INDEX_PATH")


print(f"GOOGLE_BOOKS_API_KEY: {GOOGLE_BOOKS_API_KEY}")
print(f"ASSISTANCE_API_KEY: {ASSISTANCE_API_KEY}")
print(f"BOOKS_FILE: {BOOKS_FILE}")
print(f"BOOK_RATINGS_FILE: {BOOK_RATINGS_FILE}")
print(f"MERGED_BOOKS_FILE: {MERGED_BOOKS_FILE}")
print(f"ROW_NUMBER_TO_TEST: {ROW_NUMBER_TO_TEST}")
print(f"DATASET_PATH: {DATASET_PATH}")
print(f"INDEX_PATH: {INDEX_PATH}")
