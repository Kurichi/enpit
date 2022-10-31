#%%
# データロード
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

input_file_name = './../jwtd_v2.0/train.jsonl'
df = pd.read_json(input_file_name, orient='records', lines=True)

# データの確認
print(df.iloc[0,4])
print(df.iloc[0,5])

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# 本文の文字数を確認する
df['pre_text_len'] = df['pre_text'].progress_apply(lambda x: len(tokenizer.tokenize(x)))
df['post_text_len'] = df['post_text'].progress_apply(lambda x: len(tokenizer.tokenize(x)))

# %%
# 全データをtokenizerに通して、tensorに変換
import torch

PRE_TEXT_MAX_LENGTH = df['pre_text_len'].max()
POST_TEXT_MAX_LENGTH = df['post_text_len'].max()

encodings = []
for row in tqdm(df.itertuples(), total=df.shape[0]):
    inputs = tokenizer(row.pre_text, padding='max_length', truncation=True, max_length=PRE_TEXT_MAX_LENGTH)
    outputs = tokenizer(row.post_text, padding='max_length', truncation=True, max_length=POST_TEXT_MAX_LENGTH)
    inputs['decoder_input_ids'] = outputs['input_ids']
    inputs['decoder_attention_mask'] = outputs['attention_mask']
    inputs['labels'] = outputs['input_ids'].copy()
    inputs['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in inputs['labels']]
    inputs = {k:torch.tensor(v) for k, v in inputs.items()}
    encodings.append(inputs)