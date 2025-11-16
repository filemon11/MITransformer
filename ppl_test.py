import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# 1. Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Group into blocks (for causal LM)
block_size = 128
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {k: [concatenated[k][i:i+block_size] for i in range(0, total_length, block_size)]
              for k in concatenated.keys()}
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# 2. Define tiny 1-layer, 2-head transformer
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    n_layer=1,
    n_head=2,
    n_embd=400,
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)

# 3. Set up Trainer
metric = evaluate.load("perplexity")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to log-probs
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits, shift_labels)
    return {"perplexity": torch.exp(loss).item()}

training_args = TrainingArguments(
    output_dir="./tiny_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    save_strategy="no",
    learning_rate=5e-4,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# 4. Train and evaluate
trainer.train()
results = trainer.evaluate()
print("Perplexity:", results["perplexity"])
