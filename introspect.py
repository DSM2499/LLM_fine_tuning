import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Introspector:
    def __init__(self, model = "gpt-4o-mini"):
        self.model = model

    def _format_prompt(self, task_history):
        task_description = ""
        for idx, task in enumerate(task_history, 1):
            task_description += f"Task {idx}:\nPrompt:\n{task['prompt']}\n\nResponse:\n{task['response']}\n\n"

        prompt =  f"""
You are an expert LLM fine-tuning advisor.

You are reviewing the last 5 tasks completed by a local LLM. Each task consists of a prompt and the model's response. Based on your analysis of accuracy, coherence, relevance, repetition, and instruction following, provide feedback for improvement.

Then, based on this feedback, suggest fine-tuning changes in JSON format with the following structure:

```json
{{
  "suggestions": {{
    "hyperparameters": {{
      "learning_rate": "float (e.g. 1e-5)",
      "batch_size": "int",
      "num_epochs": "int"
    }},
    "lora": {{
      "rank": "int",
      "target_modules": ["list of module names"]
    }},
    "training_focus": "describe what kind of examples to add or remove",
    "observations": ["list of high-level reflections or critiques"]
  }}
}}
Now, analyze the following tasks:

{task_description}
"""
        return prompt
    
    def run(self, task_buffer):
        prompt = self._format_prompt(task_buffer)
        response = client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "system", "content": "You are a helpful AI fine-tuning advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.3,
            )
        
        return response.choices[0].message.content