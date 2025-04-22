import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.utils import resample
import time
import csv
from tqdm import tqdm

num_file = 100

tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
vocab_size = len(tokenizer.vocab)
print("사전 크기:", vocab_size)

def encode_text(text):
    return tokenizer(text,
                     max_length=128,
                     padding="max_length",
                     truncation=True)['input_ids']

start_time = time.time()

# 파일별 인코딩 및 저장
with tqdm(total=num_file, desc="Encoding and saving files", unit="file") as pbar_file:
    for i in range(1, num_file + 1):
        try:
            df = pd.read_csv(f'data/data/data_split_{i}.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE,
                             on_bad_lines='skip').dropna()
            
            df['text'] = df['text'].apply(lambda x: x.lower())
            texts = df['text'].values
            labels = df['label'].values

            encoded_texts = []
            encoded_labels = []

            # 각 파일 내 데이터 인코딩 진행률 표시
            with tqdm(total=len(texts), desc=f"Encoding data in file {i}", unit="text", leave=False) as pbar_text:
                for text, label in zip(texts, labels):
                    try:
                        encoded_texts.append(encode_text(text))
                        encoded_labels.append(label.astype(np.float32))
                    except Exception as e:
                        print(f"Error encoding text: {text}, Error: {e}")
                    pbar_text.update(1)

            with open(f'data/encoded_data EXAONE 3.0/{i}.npz', 'wb') as f:
                np.savez_compressed(f, encoded_texts=encoded_texts, encoded_labels=encoded_labels)
            
            pbar_file.update(1)
            
            # 예상 소요 시간 업데이트
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (num_file - i) / i
            pbar_file.set_postfix({"ETA": f"{remaining_time/60:.2f}min", "Elapsed": f"{elapsed_time/60:.2f}min"})
        
        except Exception as e:
            print(f"Error processing file {i}: {e}")

# 통합 파일 저장 진행률 표시
with tqdm(total=num_file, desc="Saving combined file", unit="file") as pbar_save:
    encoded_texts = []
    encoded_labels = []

    for i in range(1, num_file + 1):
        try:
            with open(f'data/encoded_data EXAONE 3.0/{i}.npz', 'rb') as f:
                data = np.load(f)
                encoded_texts.extend(data['encoded_texts'])
                encoded_labels.extend(data['encoded_labels'])
            pbar_save.update(1)
        except Exception as e:
            print(f"Error loading encoded data from file {i}: {e}")

    print("저장 중")
    with open('data/encoded_data EXAONE 3.0/all_encoded_data.npz', 'wb') as f:
        np.savez_compressed(f, encoded_texts=encoded_texts, encoded_labels=encoded_labels)

print("저장완료")

tokenizer.save_pretrained('tokenizer_directory')
