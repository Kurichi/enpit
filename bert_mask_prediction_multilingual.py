from transformers import BertForMaskedLM, BertJapaneseTokenizer, pipeline

import torch
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer

# 再現性を確保するための処理
# torch.use_deterministic_algorithms(True)
# pl.seed_everything(42, workers=True)

model_name = "bert-base-multilingual-uncased"

unmasker = pipeline('fill-mask', model=model_name)
print(unmasker("こんにちは、私は[MASK]モデルです。"))