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

    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs.logits
        return predictions

    def get_next_word_probabilities(self, sentence, top_k=5):
        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence)

        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]

        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k).indices.tolist()

        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)

        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))

# Create an instance of the LMHeadModel
model = LMHeadModel("gpt2")

# Create a text input field for user input
user_input = st.text_input("Enter a sentence and press Enter:", "Hello how are")

# Create a number input field for choosing the number of predicted words (top K)
num_predictions = st.number_input("Number of Predicted Words (Top K):", min_value=1, value=5)

# Generate predictions when the user presses Enter
if user_input:
    if st.session_state.get("sentences") is None:
        st.session_state.sentences = []

    st.session_state.sentences.append(user_input)
    
    probabilities = model.get_next_word_probabilities(user_input.strip(), top_k=num_predictions)

    # Display the probabilities in a table with two columns
    st.header("Next Word Probabilities:")
    df = pd.DataFrame(probabilities, columns=["Word", "Probability"])
    st.dataframe(df)

# Display the list of entered sentences
if st.session_state.get("sentences"):
    st.header("Entered Sentences:")
    st.write(st.session_state.sentences)

# Add some instructions for the user
st.write("Enter a sentence, choose the number of predicted words (top K), and press Enter to see the probabilities.")
