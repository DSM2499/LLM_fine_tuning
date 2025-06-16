# 🔧 Fine-Tuning LLMs on Python & AI/ML Knowledge

This project fine-tunes a pre-trained large language model on domain-specific questions and answers related to **Python programming, machine learning, and AI concepts**. It uses **parameter-efficient fine-tuning (LoRA)** to enable training even on consumer hardware like the **MacBook Pro M2**.

---

## 📌 Objective

To improve a base LLM’s performance on:
- Answering Python and ML-related questions
- Providing better explanations of technical concepts
- Responding with more precision to parameter tuning and AI workflow queries

---

## 🧱 Project Structure

| File | Purpose |
|------|---------|
| `data/generate_dataset.py` | Generates a high-quality Q&A dataset using GPT-4o |
| `data/training_data.json` | Output dataset used for fine-tuning |
| `fine_tune.py` | Fine-tunes the LLM with LoRA on the generated dataset |

---

## 📂 Dataset Details

The dataset consists of JSONL-formatted entries like:

```json
{
  "prompt": "What is learning rate in gradient descent?",
  "completion": "The learning rate controls the size of the steps taken to reach the minimum of the loss function..."
}
```
### Topics covered:
- Python syntax, idioms, debugging

---

## ⚙️ Setup Instructions

1. Clone and setup environment
   ```bash
   git clone https://github.com/your-username/llm-fine-tune-ai.git
   cd llm-fine-tune-ai
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Generate training data (optional)
   ```bash
   python data/generate_dataset.py
   ```
3. Run fine-tuning
   ```bash
   python fine_tune.py
   ```
   By default, this will load `tiiuae/falcon-7b-instruct`, apply LoRA adapters, and fine-tune it on `data/training_data.json`.

---

## 🧠 Model & Method
- Base Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Training Method: LoRA with Hugging Face `peft`
- Training Framework: Hugging Face `Trainer`

 ---

 ## ✅ Results
 
Post-training, the model gives improved, domain-specific answers for technical queries:

"What is a learning rate?"

**Before fine-tuning**:
"A rate of learning for a model..."

**After fine-tuning**:
"The learning rate determines the size of the update step during gradient descent. Too high can overshoot minima, too low can slow convergence."

 

