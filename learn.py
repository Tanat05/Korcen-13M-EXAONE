import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import time
import os

print("TensorFlow Version:", tf.__version__)

NPZ_FILE_PATH = 'data/encoded_data EXAONE 3.0/all_encoded_data.npz'
TOKENIZER_DIR = 'tokenizer_directory'  
MODEL_SAVE_PATH = 'abusive_language_model_exaone_based.h5'


VOCAB_SIZE = 50000 
MAX_LENGTH = 128 
EMBEDDING_DIM = 128 
LSTM_UNITS = 64 
CONV_FILTERS = 128
KERNEL_SIZE = 5 
DENSE_UNITS = 64
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4

BATCH_SIZE = 64
EPOCHS = 10 
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

print(f"Loading tokenizer from {TOKENIZER_DIR}...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    VOCAB_SIZE = len(tokenizer.vocab)
    print(f"Successfully loaded tokenizer. Vocabulary Size: {VOCAB_SIZE}")
    del tokenizer
except Exception as e:
    print(f"Please ensure the tokenizer was saved correctly or manually set VOCAB_SIZE.")
    if VOCAB_SIZE == 50000: 
         print("Using default VOCAB_SIZE=50000. THIS MIGHT BE INCORRECT.")

print(f"Loading pre-encoded data from {NPZ_FILE_PATH}...")
start_time = time.time()
try:
    with np.load(NPZ_FILE_PATH) as data:
        encoded_texts = data['encoded_texts']
        encoded_labels = data['encoded_labels']
    print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
    print(f"Number of samples: {len(encoded_texts)}")
    if len(encoded_texts) == 0:
        raise ValueError("Loaded data is empty. Check the NPZ file.")
    if len(encoded_texts) != len(encoded_labels):
         raise ValueError("Mismatch between number of texts and labels.")

    encoded_texts = np.array(encoded_texts, dtype=np.int32)
    encoded_labels = np.array(encoded_labels, dtype=np.float32)

except FileNotFoundError:
    print(f"Error: NPZ file not found at {NPZ_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data from NPZ file: {e}")
    exit()

print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    encoded_texts,
    encoded_labels,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE,
    stratify=encoded_labels
)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print("Creating TensorFlow Datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("TensorFlow Datasets created.")



print("Building the model...")

# Option 1: Bidirectional LSTM based model
# model = Sequential([
#     Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH, mask_zero=True),
#     Bidirectional(LSTM(LSTM_UNITS)),
#     Dense(DENSE_UNITS, activation='relu'),
#     Dropout(DROPOUT_RATE),
#     Dense(1, activation='sigmoid')
# ])

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    Conv1D(filters=CONV_FILTERS, kernel_size=KERNEL_SIZE, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(DENSE_UNITS, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])



print("Compiling the model...")
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss',    
    patience=3,           
    verbose=1,
    restore_best_weights=True
)

MODEL_SAVE_PATH = 'abusive_language_model_exaone_based.h5'
model_checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH, 
    monitor='val_loss',
    save_best_only=True,  
    verbose=1
)


print("Starting training...")
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, model_checkpoint]
)

training_time = time.time() - start_time
print(f"\nTraining finished in {training_time / 60:.2f} minutes.")


print("\nEvaluating the best model on the validation set...")
loss, accuracy = model.evaluate(val_dataset, verbose=0)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")


print("\nModel training and evaluation complete.")
print(f"The best model weights were saved to: {MODEL_SAVE_PATH}")

try:
    import matplotlib.pyplot as plt

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
except ImportError:
    print("\nMatplotlib not found. Skipping plotting training history.")
except Exception as e:
    print(f"\nError plotting history: {e}")
