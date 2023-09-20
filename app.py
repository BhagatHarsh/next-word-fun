import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import pandas as pd

# Set Streamlit app title and set page configuration
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create a Streamlit app title with a catchy header
st.title("Next Word Predictor with GPT-2")

# Define the LMHeadModel class
class LMHeadModel:
    def __init__(self, model_name):
        # Initialize the model and the tokenizer.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_predictions(self, sentence, temperature=1.0):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=50,  # Maximum length of the generated text
                num_return_sequences=1,  # Number of sequences to generate
                temperature=temperature,  # Temperature parameter for sampling
            )
            predictions = outputs
        return predictions

    def get_next_word_probabilities(self, sentence, top_k=5, temperature=1.0):
        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence, temperature=temperature)

        # Decode the predictions into words.
        predicted_tokens = [self.tokenizer.decode(seq) for seq in predictions]

        # Split the generated text into words.
        words = predicted_tokens[0].split()

        # Calculate the total count of words.
        total_word_count = len(words)

        # Get the unique words and their counts.
        word_counts = {word: words.count(word) for word in set(words)}

        # Sort words by frequency in descending order.
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Calculate the probabilities of words.
        word_probabilities = [(word, count / total_word_count) for word, count in sorted_words]

        # Return the top k words, their counts, and their probabilities.
        return word_probabilities[:top_k]

# Create an instance of the LMHeadModel
model = LMHeadModel("gpt2")

# Create a text input field for user input
user_input = st.text_input("Enter a sentence and press Enter:", "Hello how are")

# Create a number input field for choosing the number of predicted words (top K)
num_predictions = st.number_input("Number of Predicted Words (Top K):", min_value=1, value=5)

# Create a number input field for setting the temperature (higher values make output more random)
temperature = st.number_input("Temperature (1.0 is default):", min_value=0.1, value=1.0)

# Generate predictions when the user presses Enter
if user_input:
    if st.session_state.get("sentences") is None:
        st.session_state.sentences = []

    st.session_state.sentences.append(user_input)

    # Get next word probabilities with the specified temperature
    word_probabilities = model.get_next_word_probabilities(user_input.strip(), top_k=num_predictions, temperature=temperature)

    # Display the word counts and their probabilities in a table with three columns
    st.header("Next Word Counts and Probabilities:")
    df = pd.DataFrame(word_probabilities, columns=["Word", "Count", "Probability"])
    st.dataframe(df)

# Display the list of entered sentences
if st.session_state.get("sentences"):
    st.header("Entered Sentences:")
    st.write(st.session_state.sentences)

# Add some instructions for the user
st.write("Enter a sentence, choose the number of predicted words (top K), and adjust the temperature to control randomness. Press Enter to see the word counts and their probabilities.")
