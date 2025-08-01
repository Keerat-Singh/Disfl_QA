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

# --- 4. Define Evaluation Metrics ---
# Load all the metrics you want to use
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
gleu_metric = evaluate.load("google_bleu")
wer_metric = evaluate.load("wer") # Word Error Rate (lower is better)

def compute_metrics_for_strings(predictions_list, references_list):
    """
    Computes ROUGE, BLEU, GLEU, and WER metrics given lists of
    decoded prediction and reference strings.

    Note: "Accuracy" is generally not used for text generation tasks because
    there can be multiple correct ways to phrase a corrected sentence.
    Metrics like ROUGE, BLEU, and WER are better suited for evaluating the quality
    and similarity of generated text.
    """
    results = {}
    
    # 1. Compute ROUGE scores
    rouge_result = rouge_metric.compute(
        predictions=predictions_list,
        references=references_list,
        use_stemmer=True
    )
    for key, value in rouge_result.items():
        results[key] = value.mid.fmeasure * 100 if hasattr(value, 'mid') else value * 100
    
    # 2. Compute BLEU score
    # BLEU expects references as a list of lists (e.g., [['ref1'], ['ref2']])
    bleu_result = bleu_metric.compute(
        predictions=predictions_list, 
        references=[[ref] for ref in references_list]
    )
    results['bleu'] = bleu_result['bleu'] * 100

    # 3. Compute GLEU score
    # GLEU also expects references as a list of lists
    gleu_result = gleu_metric.compute(
        predictions=predictions_list, 
        references=[[ref] for ref in references_list]
    )
    results['gleu'] = gleu_result['google_bleu'] * 100
    
    # 4. Compute WER (Word Error Rate) score
    wer_result = wer_metric.compute(
        predictions=predictions_list,
        references=references_list
    )
    # WER is a percentage, so multiply by 100 for consistency. Lower WER is better.
    results['wer'] = wer_result * 100 

    # Round all values for clean output
    final_results = {k: round(v, 4) for k, v in results.items()}
    return final_results

print("\nEvaluation metrics function 'compute_metrics_for_strings' defined.")


print(f"\n--- Loading and Testing the Model with data/test.json ---")

# --- 1. Load the fine-tuned Model and Tokenizer ---
final_model_path = "./model/flan_t5/final_model"

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
# NOTE: To check for overfitting, you should run this script once with 'data/dev.json'
# and once with 'data/test.json' and compare the results.
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
# Call the updated function with all the new metrics
evaluation_results = compute_metrics_for_strings(all_decoded_preds, all_decoded_labels)
print("\n--- Evaluation Results on Test Set ---")
print(evaluation_results)

# --- 6. Save Evaluation Results to a JSON file ---
def save_results(results_dict, output_dir="results", filename="flan_t5_evaluation_results.json"):
    """
    Saves a dictionary of evaluation results to a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Evaluation results successfully saved to: {output_path}")
    except IOError as e:
        print(f"Error: Could not save results to file '{output_path}'. Reason: {e}")

save_results(evaluation_results)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU cache cleared.")
