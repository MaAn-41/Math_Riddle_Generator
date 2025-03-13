import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, create_optimizer
import os

# ---------------------- Preparing Data ----------------------
# Step 1: Create dataset of riddles
riddles = [
    {"text": "Riddle: What number becomes zero when you subtract 15 from half of it? Solution: 30"},
    {"text": "Riddle: I'm a number whose digits sum to 14. If you add 7 to me, I become a perfect square. What number am I? Solution: 49"},
    {"text": "Riddle: I'm thinking of a number. If you multiply it by 3, then add 12, you get twice the original number. What's my number? Solution: 12"},
    {"text": "Riddle: What's the smallest positive number that, when multiplied by 3, gives a result with exactly the same digits in a different order? Solution: 1428"},
    {"text": "Riddle: I'm a two-digit number. The tens digit is one less than the ones digit. Their product is 12. What number am I? Solution: 34"},
    {"text": "Riddle: If you add me to myself, then multiply by 4, then add 8, you get 100. What number am I? Solution: 23"},
    {"text": "Riddle: I'm a number less than 50. If you divide me by 2, 3, 4, 5, or 6, you always get a remainder of 1. What number am I? Solution: 61"},
    {"text": "Riddle: The product of my digits is 36. If you add my digits, you get 13. What number am I? Solution: 94"},
    {"text": "Riddle: I'm a three-digit number. My tens digit is twice my hundreds digit. My ones digit is twice my tens digit. What number am I? Solution: 124"},
    {"text": "Riddle: Take a two-digit number. The digit in the tens place is one more than the digit in the ones place. The product of the digits is 12. What's the number? Solution: 43"},
    {"text": "Riddle: I'm a number. If you multiply me by 2 and subtract 7, you get my square root. What number am I? Solution: 9"},
    {"text": "Riddle: I'm a three-digit number. My hundreds digit is 3 less than my tens digit. My ones digit is the sum of my hundreds and tens digits. What number am I? Solution: 159"},
    {"text": "Riddle: When I'm divided by 7, the remainder is 3. When I'm divided by 5, the remainder is also 3. I'm less than 50. What number am I? Solution: 38"},
    {"text": "Riddle: I'm a number between 20 and 30. The sum of my digits is 7. If you subtract 9 from me, the result is divisible by 7. What number am I? Solution: 25"},
    {"text": "Riddle: I'm a two-digit number. If you reverse my digits, I increase by 63. What number am I? Solution: 27"},
    {"text": "Riddle: The sum of two consecutive odd numbers is 32. What's the product of these numbers? Solution: 255"},
    {"text": "Riddle: I'm a four-digit number. My thousands digit is twice my tens digit. My hundreds digit is half my ones digit. All my digits are even. What number am I? Solution: 4286"},
    {"text": "Riddle: I'm a square number between 100 and 200. The sum of my digits is 13. What number am I? Solution: 169"},
    {"text": "Riddle: I'm a number that gives the same result whether you add 5 to me or multiply me by 5. What number am I? Solution: 1.25"},
    {"text": "Riddle: I'm a two-digit prime number. If you add my digits, you get 10. What number am I? Solution: 73"},
    {"text": "Riddle: I'm a two-digit number. If you square me, then subtract 20, you get 10 times my original value. What number am I? Solution: 5"},
    {"text": "Riddle: I'm a three-digit number. My hundreds digit is twice my tens digit. My ones digit is three times my hundreds digit. What number am I? Solution: 216"},
    {"text": "Riddle: I'm a fraction. If you add 1 to my numerator, I become 1/3. If you add 2 to my denominator, I become 1/4. What fraction am I? Solution: 2/5"},
    {"text": "Riddle: The difference between a two-digit number and the number formed by reversing its digits is 54. If the sum of the digits is 12, what's the original number? Solution: 84"},
    {"text": "Riddle: I'm a number whose square plus its double equals 48. What number am I? Solution: 6"},
    {"text": "Riddle: The sum of three consecutive integers is 51. What is the product of these three integers? Solution: 4165"},
    {"text": "Riddle: I'm a three-digit number. My digits are all different. My hundreds digit is 4 more than my ones digit, and my tens digit is half my hundreds digit. What number am I? Solution: 642"},
    {"text": "Riddle: I'm a two-digit number. My tens digit is one-third of my ones digit. The sum of my digits is 12. What number am I? Solution: 39"},
    {"text": "Riddle: If you multiply me by 5, then subtract 4, you get twice my square. What number am I? Solution: 4"},
    {"text": "Riddle: I'm a two-digit number less than 40. The product of my digits is 8. What number am I? Solution: 24"}
]

# Save the riddles to a text file
with open('math_riddles.txt', 'w') as f:
    for riddle in riddles:
        f.write(riddle['text'] + '\n')

# Load model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Ensure tokenizer has required tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
def load_dataset(file_path, tokenizer, block_size=128):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenized_text = tokenizer.encode(text)
    examples = [tokenized_text[i:i + block_size] for i in range(0, len(tokenized_text) - block_size, block_size)]
    return tf.data.Dataset.from_tensor_slices(examples)

dataset = load_dataset('math_riddles.txt', tokenizer)

def add_labels(examples):
    attention_mask = [1] * len(examples)
    return {"input_ids": examples, "attention_mask": attention_mask, "labels": examples}

dataset = dataset.map(add_labels)

# Batch the dataset
batch_size = 2
dataset = dataset.batch(batch_size)

# ---------------------- Training Model ----------------------
# Set up training parameters
training_args = {
    'learning_rate': 3e-5,
    'num_train_epochs': 3,
    'weight_decay': 0.01,
}

total_steps = len(dataset) * training_args['num_train_epochs']
warmup_steps = int(0.1 * total_steps)

optimizer, lr_schedule = create_optimizer(
    init_lr=training_args['learning_rate'],
    num_train_steps=total_steps,
    num_warmup_steps=warmup_steps,
    weight_decay_rate=training_args['weight_decay'],
)

model.compile(optimizer=optimizer)

# Train the model
model.fit(dataset, epochs=training_args['num_train_epochs'])

# Save the model
model.save("riddle.keras")

# ---------------------- Generating Riddles ----------------------
def generate_riddle(model, tokenizer, prompt="Riddle: ", max_length=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    attention_mask = tf.ones_like(input_ids)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=0.8,
        top_k=50,
        do_sample=True,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate 5 test riddles
print("Generating test riddles...")
for i in range(5):
    riddle = generate_riddle(model, tokenizer)
    print(f"Test Riddle {i+1}: {riddle}")
    print("-" * 50)