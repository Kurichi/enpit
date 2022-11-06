import logging
from bert_proofreader import BertProofreader

PRETRAINED_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'
proofreader = BertProofreader(PRETRAINED_MODEL)

logging.basicConfig(level=logging.DEBUG)
print('入力受付開始')

while True:
  input_str = input().split(', ')
  print(input_str)

  if len(input_str) == 3:
    if input_str[0] == 'topk':
      proofreader.check_topk(input_str[1], topk=int(input_str[2]))
    elif input_str[0] == 'threshold':
      proofreader.check_threshold(input_str[1], threshold=float(input_str[2]))
    else:
      print('入力に誤りがあります。')
