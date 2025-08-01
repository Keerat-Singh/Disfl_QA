# test_inference.py

# --- 0. Install and Import Libraries ---
import os
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset # Import Dataset for converting pandas DataFrame
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate # Import evaluate for metrics

print("Libraries loaded successfully!")

# --- Helper function to load Disfl_QA data ---
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
            
            if original_q and disfluent_q:
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
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128
PREFIX = "correct disfluency: "

# --- 4. Define Evaluation Metrics (Simplified for direct string input) ---
rouge_metric = evaluate.load("rouge")

# This version of compute_metrics directly accepts decoded strings
def compute_metrics_for_strings(predictions_list, references_list):
    """
    Computes ROUGE metrics given lists of decoded prediction and reference strings.
    """
    # Compute ROUGE scores
    raw_rouge_result = rouge_metric.compute(
        predictions=predictions_list,
        references=references_list,
        use_stemmer=True
    )

    processed_results = {}
    for key, value in raw_rouge_result.items():
        if hasattr(value, 'mid') and hasattr(value.mid, 'fmeasure'):
            processed_results[key] = value.mid.fmeasure * 100
        elif isinstance(value, (float, np.floating)):
            processed_results[key] = value * 100
        else:
            print(f"WARNING: Unexpected ROUGE result format for key '{key}': {type(value)} - {value}")
            processed_results[key] = 0.0

    final_results = {k: round(v, 4) for k, v in processed_results.items()}
    return final_results

print("\nEvaluation metrics function 'compute_metrics_for_strings' defined.")


print(f"\n--- Loading and Testing the Model with data/test.json ---")

# --- 1. Load the fine-tuned Model and Tokenizer ---
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

# --- 3. Perform Inference and Collect Decoded Strings ---
print("\nPerforming inference on test dataset and collecting decoded predictions and labels...")
all_decoded_preds = []
all_decoded_labels = []

loaded_model.eval() # Set model to evaluation mode

batch_size = 8 # Adjust based on your GPU memory
for i in range(0, len(test_dataset), batch_size):
    batch_examples = test_dataset[i:i + batch_size]
    
    # Prepare inputs for the model
    input_texts = [PREFIX + q for q in batch_examples['disfluent_question']]
    inputs = loaded_tokenizer(
        input_texts,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length" # Ensure padding for batching
    )
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    with torch.no_grad():
        generated_ids = loaded_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode predictions
    decoded_preds_batch = loaded_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    all_decoded_preds.extend(decoded_preds_batch)

    # Decode ground truth labels (original questions)
    # No need for tokenizing labels here, just use the original strings
    all_decoded_labels.extend(batch_examples['original_question'])

print("Inference complete. Calculating metrics...")

# --- 5. Calculate Metrics ---
# Call the simplified compute_metrics_for_strings
evaluation_results = compute_metrics_for_strings(all_decoded_preds, all_decoded_labels)
print("\n--- Evaluation Results on Test Set ---")
print(evaluation_results)