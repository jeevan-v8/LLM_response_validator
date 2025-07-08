import json
from validator import get_response
from utils import calculate_similarity

def test_llm_responses():
    with open("prompts.json") as f:
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
        assert score > 0.6, f"Low similarity for prompt: {prompt}"

if __name__ == "__main__":
    test_llm_responses()