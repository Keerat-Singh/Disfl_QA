# Disfl_QA

## Project Overview
This project focuses on the task of disfluency correction, an important step in preparing spoken language for downstream NLP tasks. Using the Disfl-QA dataset, we fine-tuned several state-of-the-art Transformer-based sequence-to-sequence (Seq2Seq) models to automatically convert disfluent questions into fluent, coherent text.

The primary goal was to evaluate and compare the performance of different models to identify the most effective solution for this task. The code for the project resides here, while the large, fine-tuned models are hosted separately on the Hugging Face Hub.

## Methodology
Dataset
The project utilizes the Disfl-QA dataset, which consists of paired disfluent and fluent questions. The dataset was split into training, validation, and test sets to ensure a robust and unbiased evaluation of the models' performance.

## Models
The following models were fine-tuned for the disfluency correction task:

- BART: A denoising autoencoder model from Facebook AI.
- T5: The "Text-to-Text Transfer Transformer" from Google, which frames all NLP tasks as text-to-text problems.
- Flan-T5: A T5 model that has been further fine-tuned on a vast collection of tasks, giving it strong zero-shot and few-shot capabilities.

## Training and Evaluation
The models were trained using a standard Seq2Seq setup with the prefix correct disfluency: . The performance of each model was evaluated using a comprehensive suite of metrics to measure the quality of the generated text.

### Evaluation Metrics
The following metrics were used to evaluate the models:

- ROUGE (Recall-Oriented Understudy for Gisting Evaluation): A set of metrics that measure the overlap of n-grams between the generated and reference texts.
- BLEU (Bilingual Evaluation Understudy): A precision-focused metric for evaluating text generation.
- GLEU (Google-BLEU): An improved version of BLEU that balances precision and recall.
- WER (Word Error Rate): An error rate metric that measures the number of word edits (substitutions, insertions, deletions) needed to transform the generated text into the reference text. A lower WER indicates better performance.

## Results
The following table presents the final evaluation scores for each model on the test dataset.

| Model     | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU    | GLEU    | WER (%) |
|-----------|---------|---------|---------|---------|---------|---------|
| **BART** | 95.47   | 91.48   | 94.41   | 89.12   | 88.01   | 9.73    |
| **Flan-T5** | **96.11** | **92.44** | **95.09** | **90.34** | **89.05** | **8.96** |
| **T5** | 95.93   | 92.06   | 94.89   | 89.88   | 88.69   | 9.09    |

Note: All scores are presented as percentages (%).

Based on these results, the Flan-T5 model demonstrated the best overall performance across all key metrics.

## How to Use the Fine-Tuned Models
The fine-tuned model checkpoints are hosted on the Hugging Face Hub to handle large file sizes. You can easily download and use them in your own projects with the transformers library.

1. Install the Library:

```
pip install transformers
```

2. Load the Model and Tokenizer:
Replace [model_name] with the actual name of your repository on the Hugging Face Hub (e.g., Galmieux/t5_disfl_qa).
List of models:
- Galmieux/bart_disfl_qa
- Galmieux/flan_t5_disfl_qa
- Galmieux/t5_disfl_qa
- 
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "[YOUR_HF_REPO]"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

## Example Usage
```
text_to_correct = "I, uh, want to know how many, like, planets are there."
inputs = tokenizer(text_to_correct, return_tensors="pt")
outputs = model.generate(inputs.input_ids)
corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Corrected text: {corrected_text}")
```

Future Work and Recommendations
This project serves as a strong foundation for a robust disfluency correction system. For future development, the following steps are recommended:

Model Training: The Pegasus model was not trained due to compute constraints. The best approach for training this or any larger model would be to use a cloud-based service like Google Cloud Vertex AI or AWS SageMaker, which provides access to more powerful and scalable GPU resources.

Containerization: To ensure the project is fully reproducible and easily deployable, all code and dependencies should be packaged into a Docker container.

Advanced Evaluation: Incorporate human evaluation to get a more nuanced understanding of the generated text quality, especially for more complex disfluencies where multiple corrections may be valid.
