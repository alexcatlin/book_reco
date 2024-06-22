import os


from dotenv import load_dotenv

load_dotenv()

GOOGLE_BOOKS_API_KEY=os.environ.get("GOOGLE_BOOKS_API_KEY")
ASSISTANCE_API_KEY = os.getenv('ASSISTANCE_API_KEY')

