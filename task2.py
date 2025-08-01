import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import torch
print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
import matplotlib.pyplot as plt
import time
import os
import psutil

# 1) Train a language model using GPT2
def train_language_model(data_path, model_save_path="gpt2-jokes", epochs=1, block_size=128, batch_size=4):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=block_size,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=model_save_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Training started...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save loss per epoch
    losses = trainer.state.log_history
    loss_list = [l['loss'] for l in losses if 'loss' in l]

    return loss_list, training_time


# 2) Report hardware specifications
def report_hardware_specs():
    cpu = os.cpu_count()
    ram = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    return {"CPU Cores": cpu, "RAM (GB)": ram, "GPU": gpu}


# 3) Plot the training loss over epochs
def plot_training_loss(loss_list):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


# 4) Generate at least 5 new jokes
def generate_jokes(model_path="gpt2-jokes", prompts=None, num_jokes=5, max_length=50):
    if prompts is None:
        prompts = ["my dog", "why did", "I told my friend", "my boss", "yesterday I saw"]

    generator = pipeline("text-generation", model=model_path, tokenizer=model_path)

    jokes = []
    for prompt in prompts[:num_jokes]:
        outputs = generator(prompt, max_length=max_length, num_return_sequences=1)
        jokes.append(outputs[0]['generated_text'])

    return jokes
if __name__ == "__main__":
    # Step 1: Train the model
    losses, train_time = train_language_model("jokes.txt", epochs=10)

    # Step 2: Report hardware specs and training time
    hardware = report_hardware_specs()
    print("Hardware:", hardware)
    print(f"Training Time (seconds): {train_time:.2f}")

    # Step 3: Plot training loss
    plot_training_loss(losses)

    # Step 4: Generate jokes
    jokes = generate_jokes()
    print("\nGenerated Jokes:")
    for i, joke in enumerate(jokes, 1):
        print(f"{i}. {joke}\n")
