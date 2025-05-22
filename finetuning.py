from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from datasets import load_dataset

dataset = load_dataset("json", data_files="instruction_data.jsonl", split="train")

model_name='meta-llama/Llama-3.2-1B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # For padding

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model (FP16 or 4-bit)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Remove for full LoRA (non-quantized)
    torch_dtype=torch.float16,
    device_map="auto",
).to('cuda')


from transformers import AutoModelForCausalLM, AdapterConfig
from peft import get_peft_model

# 1. Define Adapter Configuration
adapter_config = AdapterConfig(
    adapter_type="parallel",  # or "bottleneck" (default)
    reduction_factor=16,      # hidden_size -> adapter dim= model_hidden_size//reduction_factor
    non_linearity="relu",     # Activation for adapter layers
    original_ln_before=True,  # LayerNorm before adapter
    original_ln_after=True,   # LayerNorm after adapter
    residual_before_ln=True,  # Residual connection before LayerNorm
    adapter_residual_before_ln=False,
    ln_before=False,
    ln_after=False,
    mh_adapter=True,         # Adapter for multi-head attention
    output_adapter=True,     # Adapter for output layer
    cross_adapter=False       # For cross-attention (decoder-only)
)

# 2. Apply to Model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, adapter_config)

# 3. Verify
model.print_trainable_parameters()  # Only adapters are trainable

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,            # Rank (smaller = fewer trainable params)
    lora_alpha=32,   # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Modules to adapt (depends on model)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  # For causal language modeling
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Check trainable params (~0.1% of total)


from transformers import DataCollatorForSeq2Seq
def tokenize_function(examples):
    text = [f"Instruction: {i}\nInput: {inp}\nOutput: {out}" 
            for i, inp, out in zip(examples['instruction'], 
                                  examples['input'], 
                                  examples['output'])]
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        text,
        max_length=256,          # Truncate longer sequences
        padding="max_length",    # Pad all to max_length
        truncation=True,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone() 
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

print("Sample raw data:", dataset[0])
print("Tokenized sample:", tokenize_function(dataset[:1]))
print(tokenized_dataset[0].keys())

from transformers import TrainingArguments, Trainer

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    fp16=True,  # Mixed precision
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("fine_tuned_lora_model")

from transformers import pipeline

# Load fine-tuned model
model_path = "fine_tuned_lora_model"
pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer)

# Test instruction following
instruction = "Write a joke about AI"
input_text = ""
prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"

result = pipe(prompt, max_length=128, temperature=0.7)
print(result[0]['generated_text'])
