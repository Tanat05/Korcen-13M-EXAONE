import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
import os

print("TensorFlow Version:", tf.__version__)

MODEL_LOAD_PATH = 'abusive_language_model_exaone_based.h5'
TOKENIZER_DIR = 'tokenizer_directory'
MAX_LENGTH = 128

print(f"Loading tokenizer from {TOKENIZER_DIR}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please ensure the tokenizer directory exists and is correct.")
    exit()

print(f"Loading model from {MODEL_LOAD_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_LOAD_PATH)
    model.summary()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the model file exists at {MODEL_LOAD_PATH} and TensorFlow version is compatible.")
    exit()

def preprocess_text(text_list, tokenizer, max_len):
    """
    Tokenizes and pads a list of text strings for model prediction.
    """
    processed_texts = [text.lower() for text in text_list]

    encoded = tokenizer(
        processed_texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='np' 
    )
    return encoded['input_ids']

def predict_abusive(text_list, model, tokenizer, max_len, threshold=0.5):
    """
    Takes a list of raw text strings, preprocesses them,
    predicts probabilities, and classifies based on a threshold.
    Returns probabilities and labels (1=abusive, 0=not abusive).
    """
    if isinstance(text_list, str):
        text_list = [text_list]

    processed_input = preprocess_text(text_list, tokenizer, max_len)

    probabilities = model.predict(processed_input)

    predictions = (probabilities >= threshold).astype(int)

    return probabilities.flatten(), predictions.flatten()

new_texts = [
    "다시방파줘",
    "지우개 소리로 고막 날아갔는데 반격을 해야겠어요",
    "쉬운지 어려운지 확인함",
    "박정희전차보다 더 성능좋지",
    "진지충 말한건데"
]

print("\n--- Predicting on New Texts ---")
probabilities, labels = predict_abusive(new_texts, model, tokenizer, MAX_LENGTH)

for i, text in enumerate(new_texts):
    label_text = "욕설 (Abusive)" if labels[i] == 1 else "정상 (Normal)"
    print(f"Text: \"{text}\"")
    print(f"  Probability (Abusive): {probabilities[i]:.4f}")
    print(f"  Predicted Label: {label_text} ({labels[i]})")
    print("-" * 20)
