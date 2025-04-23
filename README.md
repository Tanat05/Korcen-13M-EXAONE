<div align="center">
  <h1>Korcen-13M-EXAONE</h1>
  <h2>This failure, though another, is a better one.</h2>
</div>

![131_20220604170616](https://user-images.githubusercontent.com/85154556/171998341-9a7439c8-122f-4a9f-beb6-0e0b3aad05ed.png)

"Refined Intelligence: Enhanced Accuracy and Adaptability in ML Filtering (Lessons from a Setback)."

This project initially aimed to be an advanced iteration of our machine learning-based filter, leveraging a significantly larger dataset. However, the project faced a setback due to the compromised quality of this expanded data, ultimately leading to unsatisfactory filtering performance.

Undeterred by this challenge, we are committed to overcoming this data quality issue. We are actively focusing on refining our data acquisition and cleaning processes and will continue to develop and release upgraded models that progressively enhance accuracy, reduce false positives, and improve adaptability to evolving slang and offensive language. Our dedication to providing a robust and reliable filtering solution remains unwavering.

[Korcen](https://github.com/KR-korcen/korcen): original before innovation.

[Korcen-kogpt2](https://github.com/Tanat05/korcen-kogpt2): First innovation and first failure

# Model Overview
```
total samples: 14,879,960
Training samples: 11,903,968
Validation samples: 2,975,992
```

Parameters: 13,197,569

Training time: 4h

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
