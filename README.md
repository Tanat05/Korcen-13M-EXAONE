<div align="center">
  <h1>Korcen-13M-EXAONE</h1>
  <h2>This failure, though another, is a better one.</h2>
</div>

![131_20220604170616](https://user-images.githubusercontent.com/85154556/171998341-9a7439c8-122f-4a9f-beb6-0e0b3aad05ed.png)

Korcen-13M-EXAONE is an AI model developed with the primary purpose of filtering inappropriate language in social media (SNS) chats. Currently, the model's performance is affected by limitations in the quality of its training data. However, future efforts will concentrate on acquiring and refining high-quality data to significantly enhance its accuracy in detecting Korean profanity.

Key Features:
- **Designed for SNS Chat Filtering**: Specifically built to identify and filter out offensive language within the context of social media conversations.
- **Korean Language Specific**: Engineered with an understanding of various Korean profanity expressions to achieve precise detection.
  
Future Development:
- **Continuous Improvement via Data Quality**: The immediate focus is on improving the model's performance by enhancing the quality of its training data.
- **Enhanced Filtering Based on Conversation Flow**: Once Korcen-13M-EXAONE achieves a high level of accuracy, the development of a more advanced model capable of understanding the flow of conversations to filter inappropriate content within context is planned.
  
Korcen-13M-EXAONE represents an initial step towards creating a safer and more positive online communication environment. The ongoing commitment to data quality improvement will pave the way for more sophisticated content filtering solutions in the future.

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
