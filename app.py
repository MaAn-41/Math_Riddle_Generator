import streamlit as st
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)
model.load_weights("riddle.keras")

st.title("AI-Powered Math Riddle Generator")
st.write("Generate fun and challenging math riddles using AI!")

def generate_riddle(prompt="Riddle: ", max_length=100, temperature=0.7):
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

prompt_text = st.text_input("Enter a custom riddle prompt:", "Riddle: ")

if st.button("Generate Riddle"):
    with st.spinner("Generating..."):
        riddle = generate_riddle(prompt_text)
        st.success(riddle)
