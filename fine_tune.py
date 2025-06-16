import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float32)
model.to(device)

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training Data
dataset = load_dataset("json", data_files = "data/full_dataset.json", split = "train")

tokenized = dataset.map(
    lambda ex: tokenizer(
        f"### Question:\n{ex['prompt']}\n\n### Answer:\n{ex['response']}",
        padding = "max_length",
        truncation = True,
        max_length = 256
    ),
    remove_columns = ["prompt", "response"]
)

#Training
training_args = TrainingArguments(
    output_dir = "./checkpoints",
    per_device_train_batch_size = 1,
    num_train_epochs = 2,
    learning_rate = 1e-5,
    logging_steps = 10,
    save_strategy = "epoch",
    report_to = "none",
    bf16 = False,
    fp16 = False
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized,
    tokenizer = tokenizer,
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)
)

print("🚀 Starting fine-tuning with TinyLlama...")
trainer.train()
print("✅ Training completed. Model saved to ./checkpoints")