# --- 0. Install and Import Libraries ---
import os
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset # Import Dataset for converting pandas DataFrame
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.data.data_collator import DataCollatorForSeq2Seq
import evaluate

# --- 1. Load the Disfl_QA Dataset from local JSON files ---

def load_disfl_qa_data(filepath):
    """
    Loads the Disfl-QA dataset from a JSON file and extracts
    'original' and 'disfluent' question pairs.

    Args:
        filepath (str): The path to the train.json or test.json file.

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
            else:
                pass # Skip logging to avoid too much output if this is expected

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

# Define your local data paths
# Adjust these paths based on where your train.json and test.json are located
train_filepath = 'data/train.json' 
test_filepath = 'data/test.json'     

# Ensure the data directory exists
if not os.path.exists('data'):
    print("Error: 'data' directory not found. Please create it and place train.json/test.json inside.")
    exit()

print(f"Loading training data from: {train_filepath}")
train_data_list = load_disfl_qa_data(train_filepath)

print(f"Loading validation data from: {test_filepath}")
eval_data_list = load_disfl_qa_data(test_filepath)


if train_data_list:
    print(f"Successfully loaded {len(train_data_list)} training question pairs.")
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data_list))
else:
    print("No training data loaded. Please check the file path and content.")
    exit() # Exit if no training data to prevent errors

if eval_data_list:
    print(f"Successfully loaded {len(eval_data_list)} validation question pairs.")
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data_list))
else:
    print("No validation data loaded. Proceeding without validation set.")
    eval_dataset = None # Set to None if no validation data is loaded

print("\nLoaded datasets:")
print("Train Dataset:", train_dataset)
if eval_dataset:
    print("Validation Dataset:", eval_dataset)

# --- 2. Initialize Tokenizer and Model ---
MODEL_CHECKPOINT = "t5-base"

print(f"\nLoading tokenizer from {MODEL_CHECKPOINT}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
print("Tokenizer loaded!")

print(f"Loading model from {MODEL_CHECKPOINT}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
print("Model loaded!")

# --- 3. Preprocessing the Dataset ---
# Task: Disfluency Correction (Disfluent Question -> Original Question)
prefix = "correct disfluency: " # A suitable prefix for this task

# Define max sequence lengths for input and target
MAX_INPUT_LENGTH = 256  # Max length for disfluent question + prefix
MAX_TARGET_LENGTH = 128 # Max length for original (fluent) question

def preprocess_function(examples):
    # Inputs are 'disfluent_question', targets are 'original_question'
    inputs = [prefix + q for q in examples["disfluent_question"]]
    targets = [q for q in examples["original_question"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print(f"\nPreprocessing dataset for disfluency correction (max_input_length={MAX_INPUT_LENGTH}, max_target_length={MAX_TARGET_LENGTH})...")

tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)
print("Training dataset preprocessing complete!")

if eval_dataset:
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    print("Validation dataset preprocessing complete!")
else:
    tokenized_eval_dataset = None

print("\nTokenized datasets:")
print("Train:", tokenized_train_dataset)
if tokenized_eval_dataset:
    print("Validation:", tokenized_eval_dataset)

# --- 4. Define Evaluation Metrics ---
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds.predictions, eval_preds.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]

    if isinstance(preds, torch.Tensor):
        predicted_token_ids = preds.argmax(dim=-1).cpu().numpy()
    elif isinstance(preds, np.ndarray):
        predicted_token_ids = np.argmax(preds, axis=-1)
    else:
        raise TypeError(f"Unexpected type for predictions: {type(preds)}")

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    elif isinstance(labels, np.ndarray):
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    else:
        raise TypeError(f"Unexpected type for labels: {type(labels)}")

    decoded_preds = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    raw_rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    processed_results = {}
    for key, value in raw_rouge_result.items():
        # Make the metric computation more robust
        # Check if 'value' is a RougeResult object (has 'mid' and 'mid.fmeasure')
        if hasattr(value, 'mid') and hasattr(value.mid, 'fmeasure'):
            processed_results[key] = value.mid.fmeasure * 100
        elif isinstance(value, (float, np.floating)):
            # If it's already a float (e.g., from an internal NaN/division by zero in evaluate)
            # just use the float directly. If it's NaN, it will remain NaN.
            processed_results[key] = value * 100
        else:
            # Fallback for truly unexpected types, should not happen normally
            print(f"WARNING: Unexpected ROUGE result format for key '{key}': {type(value)} - {value}")
            processed_results[key] = 0.0 # Default to 0, or np.nan if you prefer NaNs for uncomputable metrics

    # Round the results for cleaner output
    final_results = {k: round(v, 4) for k, v in processed_results.items()} # Corrected line
    return final_results

print("\nEvaluation metrics function 'compute_metrics' defined.")

# --- 5. Define Data Collator ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
print("\nData collator initialized.")

# --- 6. Configure Training Arguments ---
OUTPUT_DIR = "./model/t5/"
print("\nConfiguring training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    report_to="tensorboard",
    # *** IMPORTANT CHANGE FOR STABILITY: Set fp16 to False initially if you see NaNs ***
    # If fp16 causes issues, disable it. You can re-enable after stable training with optim="adafactor"
    fp16=False, # Changed from torch.cuda.is_available() for troubleshooting NaN issues
    # Consider using Adafactor optimizer for T5 for better stability and memory efficiency
    # optim="adafactor", # Uncomment this line to use Adafactor
    predict_with_generate=True,
    # push_to_hub=False,
    # hub_model_id="your-username/flan-t5-base-disfl-qa-disfluency-correction",
)
print("Training arguments configured!")

# --- 7. Initialize the Trainer ---
print("\nInitializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer, # Pass tokenizer to Trainer for generation during evaluation
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("Trainer initialized!")

# --- 8. Start Training ---
print("\nStarting training...")
trainer.train()
print("Training complete!")

# --- 9. Evaluate the Best Model ---
if tokenized_eval_dataset:
    print("\nEvaluating on the validation set after training (best model loaded)...")
    final_evaluation_results = trainer.evaluate(tokenized_eval_dataset)
    print(f"Final Validation Results: {final_evaluation_results}")
else:
    print("\nNo validation dataset provided for final evaluation.")

# --- 10. Save the Fine-tuned Model and Tokenizer ---
final_model_path = os.path.join(OUTPUT_DIR, "final_model")
print(f"\nSaving final model to {final_model_path}...")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print("Model and tokenizer saved!")

print("\nFine-tuning process complete. You can now load the model from:")
print(final_model_path)

# --- 11. Example Inference with the Fine-tuned Model ---
print("\n--- Example Inference ---")
# Example of how to load the model later for inference:
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# loaded_tokenizer = AutoTokenizer.from_pretrained(final_model_path)
# loaded_model = AutoModelForSeq2SeqLM.from_pretrained(final_model_path)

test_disfluent_question = "correct disfluency: um, what is the, uh, capital of France?"
print(f"Input: {test_disfluent_question}")

input_ids = tokenizer(
    test_disfluent_question,
    return_tensors="pt",
    max_length=MAX_INPUT_LENGTH,
    truncation=True
).input_ids

if torch.cuda.is_available():
    model.to("cuda") # Ensure model is on GPU
    input_ids = input_ids.to("cuda")

generated_ids = model.generate(input_ids, max_new_tokens=MAX_TARGET_LENGTH)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated (Corrected): {generated_text}")

test_disfluent_question_2 = "correct disfluency: uh, how, how do I, uh, open the door?"
print(f"\nInput: {test_disfluent_question_2}")
input_ids_2 = tokenizer(
    test_disfluent_question_2,
    return_tensors="pt",
    max_length=MAX_INPUT_LENGTH,
    truncation=True
).input_ids
if torch.cuda.is_available():
    input_ids_2 = input_ids_2.to("cuda")

generated_ids_2 = model.generate(input_ids_2, max_new_tokens=MAX_TARGET_LENGTH)
generated_text_2 = tokenizer.decode(generated_ids_2[0], skip_special_tokens=True)
print(f"Generated (Corrected): {generated_text_2}")