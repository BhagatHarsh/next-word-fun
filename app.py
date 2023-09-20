import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# Create a Streamlit app title
st.title("Language Model Probability Explorer")

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
user_input = st.text_input("Enter a sentence:", "Hello how are")

# Create a number input field for choosing the number of predicted words
num_predictions = st.number_input("Number of Predicted Words (Top K):", min_value=1, value=5)

# Create a button to trigger the predictions
if st.button("Get Next Word Probabilities"):
    # Get the probabilities for the user input
    probabilities = model.get_next_word_probabilities(user_input, top_k=num_predictions)

    # Display the probabilities
    st.header("Next Word Probabilities:")
    for token, prob in probabilities:
        st.write(f"- {token}: {prob:.4f}")

# Add some instructions for the user
st.write("Enter a sentence, choose the number of predicted words (top K), and click the button to see the probabilities.")
