import json
from utils import calculate_similarity
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env variables
load_dotenv()

# OPTIONAL: Print whether the API key is loaded correctly
print("API Key loaded:", bool(os.getenv("OPENAI_API_KEY")))

# Create OpenAI client
client = OpenAI()  # auto uses key from env

def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def validate_all():
    with open("prompts.json", "r") as f:
        test_data = json.load(f)

    for item in test_data:
        prompt = item["prompt"]
        expected = item["expected_answer"]

        actual = get_response(prompt)
        score = calculate_similarity(actual, expected)

        print(f"\nPrompt: {prompt}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Similarity Score: {score:.2f}")

if __name__ == "__main__":
    validate_all()