# test_inference.py

# --- 0. Install and Import Libraries ---
import os
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset # Import Dataset for converting pandas DataFrame
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

print("Libraries loaded successfully!")

# --- Helper function to load Disfl_QA data (copied from your training script) ---
def load_disfl_qa_data(filepath):
    """
    Loads the Disfl-QA dataset from a JSON file and extracts
    'original' and 'disfluent' question pairs.

    Args:
        filepath (str): The path to the train.json, dev.json, or test.json file.

    Returns:
        list of dict: A list where each dictionary contains
                      'original_question' and 'disfluent_question'.
    """
    data_pairs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for squad_id, questions in raw_data.items():
            original_q = questions.get('original')
            disfluent_q = questions.get('disfluent')
            
            if original_q and disfluent_q: # Ensure both exist
                data_pairs.append({
                    'squad_v2_id': squad_id,
                    'original_question': original_q,
                    'disfluent_question': disfluent_q
                })
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return []
    return data_pairs


# --- Define constants consistent with your training script ---
# These should match the values used during training for consistency
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128
PREFIX = "correct disfluency: " # Ensure this prefix is the same as used in training

print(f"\n--- Loading and Testing the Model with data/test.json ---")

# --- 1. Load the fine-tuned Model and Tokenizer ---
# Make sure this path points to where you saved your model in the training script
# For example, if OUTPUT_DIR was "./results_disfl_qa_disfluency_correction"
# and you saved it as "final_model", the path would be "results_disfl_qa_disfluency_correction/final_model"
final_model_path = "./model/pegasus/final_model"

if not os.path.exists(final_model_path):
    print(f"Error: Model path '{final_model_path}' does not exist.")
    print("Please ensure the path is correct and the model was saved there during training.")
    exit()

print(f"Loading tokenizer from {final_model_path}...")
loaded_tokenizer = AutoTokenizer.from_pretrained(final_model_path)
print("Tokenizer loaded!")

print(f"Loading model from {final_model_path}...")
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(final_model_path)
print("Model loaded!")

# Move model to GPU if available
if torch.cuda.is_available():
    loaded_model.to("cuda")
    print("Model moved to GPU.")
else:
    print("No GPU available, running inference on CPU.")

# --- 2. Load the test dataset ---
test_filepath = 'data/test.json'

if not os.path.exists('data'):
    print("Error: 'data' directory not found. Please create it and place test.json inside.")
    exit()

print(f"\nLoading test data from: {test_filepath}")
test_data_list = load_disfl_qa_data(test_filepath)

if test_data_list:
    print(f"Successfully loaded {len(test_data_list)} test question pairs.")
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data_list))
else:
    print("No test data loaded. Please check the file path and content.")
    exit()

print("\n--- Running Inference on Test Dataset ---")

# You can iterate through the dataset or a subset of it
# For demonstration, let's process the first few examples or all of them
num_examples_to_test = 5 # Set to None to test all, or an integer for a subset

random_indices = random.sample(range(len(test_dataset)), num_examples_to_test)
test_examples = test_dataset.select(random_indices)

for i, example in enumerate(test_examples):
    disfluent_q = example['disfluent_question']
    original_q = example['original_question'] # The ground truth for comparison

    # Prepare input for the model
    input_text = PREFIX + disfluent_q
    input_ids = loaded_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    ).input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # Generate prediction
    generated_ids = loaded_model.generate(
        input_ids,
        max_new_tokens=MAX_TARGET_LENGTH,
        num_beams=4, # Often helps with quality in generation
        early_stopping=True # Stop when all beam hypotheses have finished
    )
    generated_text = loaded_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n--- Test Example {i+1} ---")
    print(f"Disfluent Question: {disfluent_q}")
    print(f"Ground Truth (Original): {original_q}")
    print(f"Model Generated (Corrected): {generated_text}")

print("\nInference on test data complete!")