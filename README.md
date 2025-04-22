# Korcen-13M-EXAONE
Korcen-13M-EXAONE is an AI model developed for the purpose of identifying the presence of profanity within Korean text. While the model currently faces performance limitations due to data quality issues in its training set, future efforts will focus on acquiring and utilizing high-quality data to significantly improve its accuracy.

Key Features:
- Korean Language Specific: Designed with an understanding of diverse Korean profanity expressions to achieve accurate detection.
- Continuous Improvement Focus: Prioritizing the enhancement of data quality as the primary strategy for ongoing model performance improvement.

Future Plans:
- Data Quality Enhancement: A dedicated focus will be placed on acquiring and refining high-quality Korean profanity data.
- Model Retraining: The model will undergo retraining using the improved data to maximize its profanity detection capabilities.

Korcen-13M-EXAONE aims to evolve into a more robust and reliable Korean profanity detection model through continuous data quality improvements. Your interest and support are appreciated.
```
total samples: 14,879,960
Training samples: 11,903,968
Validation samples: 2,975,992
```
# Model Overview
Parameters: 13,197,569

Training time: 5h

Tokenizer: EXAONE 3.5 Tokenizer (vocab size: 102,400)

![Figure_1](https://github.com/user-attachments/assets/08abd495-f039-4f00-af54-d95001a1c05f)

# Example PY: 3.10 TF: 2.10
```py
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
import os

print("TensorFlow Version:", tf.__version__)

MODEL_LOAD_PATH = 'abusive_language_model_exaone_based.h5'
TOKENIZER_DIR = 'tokenizer_directory'
MAX_LENGTH = 128

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please ensure the tokenizer directory exists and is correct.")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_LOAD_PATH)
    model.summary()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the model file exists at {MODEL_LOAD_PATH} and TensorFlow version is compatible.")
    exit()

def preprocess_text(text, tokenizer, max_len):
    processed_text = text.lower()
    encoded = tokenizer(
        processed_text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    return encoded['input_ids']

def predict_abusive(text, model, tokenizer, max_len, threshold=0.5):
    processed_input = preprocess_text(text, tokenizer, max_len)
    probability = model.predict(processed_input)
    prediction = (probability >= threshold).astype(int)
    return probability.flatten()[0], prediction.flatten()[0]

input_text = input("Please enter a sentence: ")
probability, label = predict_abusive(input_text, model, tokenizer, MAX_LENGTH)

label_text = "욕설 (Abusive)" if label == 1 else "정상 (Normal)"
print(f"Text: \"{input_text}\"")
print(f"Probability (Abusive): {probability:.4f}")
print(f"Predicted Label: {label_text} ({label})")
```
