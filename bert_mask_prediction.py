from transformers import BertForMaskedLM, BertJapaneseTokenizer

import torch
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer

# 再現性を確保するための処理
# torch.use_deterministic_algorithms(True)
# pl.seed_everything(42, workers=True)

model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert_model = BertForMaskedLM.from_pretrained(model_name)

# bert_model = bert_model.cuda() # GPU上で使えるように変換

# text = '『神戸のデータ活用塾！KDL Data Blog』では[MASK]についての記事を発信しています。是非、一度ご覧ください。'
text = 'こんにちは、私は[MASK]モデルです。'

# 形態素に分割
tokens = tokenizer.tokenize(text)

# 形態素を対応するIDに変換
input_ids = tokenizer.encode(text, return_tensors='pt')
# input_ids = input_ids.cuda() # GPU上で使えるように変換


# モデルの出力を得る
with torch.no_grad():
  output = bert_model(input_ids=input_ids)
  scores = output.logits

# 最も確率の高い形態素を抽出
mask_position = input_ids[0].tolist().index(4)
index_sorted = scores[0, mask_position].argsort(descending = True).tolist()
sorted = scores[0, mask_position].tolist()
# print(sorted)
# sorted.sort(reverse=True)
# print(sorted[0:10])
for item in index_sorted[0:10]:
  token = tokenizer.convert_ids_to_tokens(item)
  token = token.replace('##','')
  print(f'トークン：{token}、スコア：{sorted[item]}')
# id_best = scores[0, mask_position].argmax(-1).item()
# token_best  = tokenizer.convert_ids_to_tokens(id_best)
# # print(token_best)
# token_best = token_best.replace('##', '')

# 実際にテキストを書き換える
# text = text.replace('[MASK]', token_best)
# print(text)
