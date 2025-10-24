import os
import json
import time
import requests
from tqdm import tqdm

class ReasoningPairsGenerator:
    """
    Generate detailed and condensed reasoning pairs for raw problems using ChatGPT-4o-mini.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = "api_key"):
        """
        Initialize the generator with API credentials.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key
        self.model = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def generate_reasoning(self, query: str, dataset_name:str) -> str:
        """
        Generate detailed reasoning for a given query.

        Args:
            query: The mathematical problem to solve

        Returns:
            Detailed step-by-step reasoning
        """
         # Determine the type of problem
        if dataset_name == "commonsense_qa":
            messages = [
                {"role": "system", "content": "You are an AI that provides step-by-step reasoning for multiple-choice commonsense questions."},
                {"role": "user", "content": f"Problem: {query}\n\nPlease think through this question step by step. Consider what each answer option means and why it would or wouldn't be appropriate. At the end, provide your answer as a single letter (A, B, C, D, or E)."}
            ]
        elif dataset_name == "coin_flip":
            messages = [
                {"role": "system", "content": "You are an AI that explains coin flip logical problems."},
                {"role": "user", "content": f"Problem: {query}\n\nPlease solve this problem by tracking the state of the coin at each step. You only need to consider whether the coin is heads up or tails up after each flip or non-flip. At the end, answer with 'yes' or 'no' to whether the coin is still heads up."}
            ]
        else:
            # Default math tutor prompt
            messages = [
                {"role": "system", "content": "You are a math tutor who provides detailed step-by-step solutions to math problems."},
                {"role": "user", "content": f"Problem: {query}\n\nPlease solve this step-by-step, showing all your work."}
            ]
        
        count = 0
        while True:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.2,
                    }
                )
                response.raise_for_status()
                reasoning = response.json()["choices"][0]["message"]["content"]
                return reasoning
            except Exception as e:
                count += 1
                if count == 10:
                    return ""
                print(f"Error generating detailed reasoning: {e}")
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"API response: {response.text}")
                time.sleep(0.001)

    def generate_condensed_reasoning(self, query: str, original_reasoning: str) -> str:
        """
        Generate condensed version of the original reasoning.

        Args:
            query: The original problem
            original_reasoning: The detailed reasoning to condense

        Returns:
            Condensed reasoning
        """
        messages = [
            {"role": "system", "content": "You are an AI that creates concise summaries of problem reasoning."},
            {"role": "user", "content": f"Problem: {query}\n\nDetailed Solution: {original_reasoning}\n\nPlease condense the above solution into a brief but complete chain of reasoning. Be concise but maintain accuracy within 10 words."}
        ]
        count = 0
        while True:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.2,
                    }
                )
                response.raise_for_status()
                condensed = response.json()["choices"][0]["message"]["content"]
                return condensed
            except Exception as e:
                count += 1
                if count == 10:
                    return ""
                print(f"Error generating condensed reasoning: {e}")
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"API response: {response.text}")
                time.sleep(0.001)
    
    def create_dataset(self, dataset:list[dict[str, str]], save_path:str):
        if os.path.exists(save_path):
            dataset = json.load(open(save_path, "r", encoding='utf-8'))
        for i in tqdm(range(len(dataset))):
            if "condensed_reasoning" in dataset[i]:
                continue
            try:
                dataset[i]["condensed_reasoning"] = self.generate_condensed_reasoning(dataset[i]["query"], dataset[i]["reasoning"])
                # Save progress after every 5 examples or at the last one
                if (i + 1) % 5 == 0 or i == len(dataset) - 1:
                    # Write to a temporary file first, then rename to avoid corruption
                    temp_file = f"{save_path}.temp"
                    if not os.path.exists(temp_file):
                        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
                    json.dump(dataset, open(temp_file, "w", encoding='utf-8'), indent=2, ensure_ascii=False)
                    # Safely replace the original file
                    os.replace(temp_file, save_path)
                time.sleep(0.001)
            except Exception as e:
                print(f"Error processing query {i}: {e}")
                # Save progress even on failure
                temp_file = f"{save_path}.temp"
                json.dump(dataset, open(temp_file, "w", encoding='utf-8'), indent=2, ensure_ascii=False)
                os.replace(temp_file, save_path)
        return dataset