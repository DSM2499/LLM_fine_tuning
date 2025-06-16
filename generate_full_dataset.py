import openai
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
GENERATED_PROMPT_FILE = "data/generated_prompts.json"
OUTPUT_FILE = "data/full_dataset.json"

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_prompt_list():
    system_msg = "You are an expert AI educator and curriculum designer."
    user_msg = (
        "Generate a list of 100 high-quality prompts/questions related to Python programming, AI/ML, model training, hyperparameter tuning, neural network architecture, regularization, optimization, and LLM fine-tuning. "
        "Each question should test or teach a key concept and be suitable for an AI assistant to answer in a detailed, instructional way."
    )
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature = 0.5
    )
    lines = response.choices[0].message.content.split("\n")
    prompts = [line.strip().lstrip("0123456789. ") for line in lines if line.strip()]
    return prompts

def generate_prompt_response(prompt):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a highly experienced AI/ML tutor. Respond to the prompt clearly, with technical accuracy, examples, and useful depth."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.7
    )
    return {
        "prompt": prompt,
        "response": response.choices[0].message.content.strip()
    }

def main():
    os.makedirs("data", exist_ok = True)

    # Step 1: Generate prompt list
    print("🧠 Generating 100 prompt ideas...")
    prompts = get_prompt_list()
    print(f"✅ Got {len(prompts)} prompts.")

    with open(GENERATED_PROMPT_FILE, "w") as f:
        json.dump(prompts, f, indent = 2)

    # Step 2: Generate completions
    print("💬 Generating answers with GPT-4o...")
    samples = []
    for prompt in tqdm(prompts):
        try:
            pair = generate_prompt_response(prompt)
            samples.append(pair)
        except Exception as e:
            print(f"❌ Error generating for prompt: {prompt}\n{e}")

    with open(OUTPUT_FILE, "w") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")

    print(f"\n✅ Finished. {len(samples)} samples saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()