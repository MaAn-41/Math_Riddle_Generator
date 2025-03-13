import streamlit as st
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load the fine-tuned model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("riddle.keras")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    return model, tokenizer

model, tokenizer = load_model()

def generate_riddle(prompt="Riddle: ", max_length=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
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
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("Math Riddle Generator")
st.write("Generate unique math riddles using AI!")

if st.button("Generate Riddle"):
    riddle = generate_riddle()
    st.write("### Riddle:")
    st.write(riddle)
