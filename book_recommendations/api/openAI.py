# openAI.py
# openAI.py
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

class AssistanceAPI:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        # Set pad_token to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, truncation=True)

    def clean_query(self, query):
        # Replace inappropriate language with a placeholder
        inappropriate_words = ['cunt']
        for word in inappropriate_words:
            query = query.replace(word, '[censored]')
        return query

    def get_assistance(self, query):
        cleaned_query = self.clean_query(query)
        prompt = (
            f"Can you recommend a book similar to '{cleaned_query}'? "
            "For example, if the request is 'Norwegian Wood by Haruki Murakami', the response should be 'I recommend 'Kafka on the Shore' by Haruki Murakami'. "
            "Please provide a specific book title and author."
        )
        response = self.pipeline(prompt, max_length=150, num_return_sequences=1, truncation=True)
        generated_text = response[0]['generated_text']
        
        # Debugging: Print the full generated text
        print(f"Debug: Full generated text: {generated_text}")
        
        # Post-processing: Split into sentences and return the first complete sentence that contains a book recommendation
        sentences = generated_text.split('.')
        for sentence in sentences:
            if 'recommend' in sentence.lower() or 'suggest' in sentence.lower() or 'I recommend' in sentence:
                return sentence.strip() + '.'
        # If no suitable sentence is found, return the first complete sentence
        if sentences:
            return sentences[0].strip() + '.'
        return generated_text.strip()

# Example usage
if __name__ == "__main__":
    assistance_api = AssistanceAPI()
    response = assistance_api.get_assistance("recommend me a book like Norwegian Wood by Haruki Murakami")
    print(response)
