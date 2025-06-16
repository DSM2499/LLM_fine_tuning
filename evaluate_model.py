import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import openai
import os
import pandas as pd
from dotenv import load_dotenv
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "What is the capital of Brazil?",
    "Explain how gradient descent works.",
    "Summarize the causes of climate change.",
    "Translate to French: 'Good morning!'",
    "Write a haiku about the moon."
]

def batch_score_with_gpt4(base_response, tuned_response, prompt) -> pd.DataFrame:
    """
    Score and critique base vs fine-tuned responses using GPT-4.
    """
    critique_prompt = f"""
    You are a helpful AI evaluator. Given a prompt and two answers — one from the base model and one from a fine-tuned model — assess their quality.

PROMPT: {prompt}
🔹 BASE MODEL RESPONSE:
{base_response}

🔸 FINE-TUNED MODEL RESPONSE:
{tuned_response}

Evaluate each on:
1. Factual Accuracy (1-5)
2. Clarity (1-5)
3. Conciseness (1-5)
4. Instruction Following (1-5)
5. Overall Quality (1-10)

hen, explain which model performed better and why in 2-3 sentences.

Return JSON format:
{{
  "factual_accuracy": int,
  "clarity": int,
  "conciseness": int,
  "instruction_following": int,
  "overall_quality": int,
  "winner": "base" or "tuned",
  "reasoning": "string"
}}
"""
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": "You are an expert NLP model evaluator."},
            {"role": "user", "content": critique_prompt}
        ],
        response_format = {"type": "json_object"}
    )
    result_json = json.loads(response.choices[0].message.content)
    result_json["prompt"] = prompt
    result_json["base_response"] = base_response
    result_json["tuned_response"] = tuned_response
    results_df = pd.DataFrame([result_json])
    return results_df
    

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float32)
base_pipe = pipeline("text-generation", model = base_model, tokenizer = tokenizer, device = device)

#Load fine-tuned model
ft_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float32).to(device)
ft_model = PeftModel.from_pretrained(ft_model, "checkpoints/checkpoint-224")
ft_pipe = pipeline("text-generation", model = ft_model, tokenizer = tokenizer, device = device)

# Compare outputs
print("\n📊 Evaluation Report")
results_df = pd.DataFrame()
for i, prompt in enumerate(prompts, 1):
    print(f"\n{i}. Prompt: {prompt}")
    base = base_pipe(prompt, max_new_tokens = 60)[0]["generated_text"].replace("\n", " ")
    print(f"🔹 Base: {base}")
    tuned = ft_pipe(prompt, max_new_tokens = 60)[0]["generated_text"].replace("\n", " ")
    print(f"🔸 Tuned: {tuned}")
    results_df = pd.concat([results_df, batch_score_with_gpt4(base, tuned, prompt)], ignore_index = True)

print("\n📊 Evaluation Results:")
print(results_df)




