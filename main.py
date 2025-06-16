from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

quant_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = "float16",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = quant_config,
    device_map = "auto",
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout = 0.1,
    bias = "none",
    task_type = "CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()