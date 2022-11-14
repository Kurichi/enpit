# %%
# load jwtd
import pandas as pd

input_file_name = './jwtd_v2.0/train.jsonl'
df = pd.read_json(input_file_name, orient='records', lines=True)

# %%
import CaboCha

def gen_chunks(tree):
    """
    構文木treeからチャンクの辞書を生成する
    """
    chunks = {}
    key = 0  # intにしているがこれはChunk.linkの値で辿れるようにしている

    for i in range(tree.size()):  # ツリーのサイズだけ回す
        tok = tree.token(i)  # トークンを得る
        if tok.chunk:  # トークンがチャンクを持っていたら
            chunks[key] = tok.chunk  # チャンクを辞書に追加する
            key += 1

    return chunks


def get_surface(tree, chunk):
    """
    chunkからtree内のトークンを得て、そのトークンが持つ表層形を取得する
    """
    surface = ''
    beg = chunk.token_pos  # このチャンクのツリー内のトークンの位置
    end = chunk.token_pos + chunk.token_size  # トークン列のサイズ

    for i in range(beg, end):
        token = tree.token(i)
        feature = token.feature.split(',')
        # print(feature[0])
        # if not feature[0] in ['名詞', '動詞']:
        #     return surface
        surface += token.surface  # 表層形の取得

    return surface


def make_pair(sentence, is_right):
    cp = CaboCha.Parser()  # パーサーを得る
    tree = cp.parse(sentence)  # 入力から構文木を生成
    # print(tree.toString(CaboCha.FORMAT_TREE))  # デバッグ用

    chunks = gen_chunks(tree)  # チャンクの辞書を生成する

    word_pairs = []

    for from_chunk in chunks.values():
        if from_chunk.link < 0:
            continue  # リンクのないチャンクは飛ばす

        # このチャンクの表層形を取得
        from_surface = get_surface(tree, from_chunk)

        # from_chunkがリンクしているチャンクを取得
        to_chunk = chunks[from_chunk.link]
        to_surface = get_surface(tree, to_chunk)
        word_pairs.append((from_surface, to_surface, is_right))

    return word_pairs

# %%
from tqdm import tqdm

def main():
    data = pd.DataFrame()
    with tqdm(total=len(df)) as pbar:
        for d in tqdm(df.itertuples()):
            pair = make_pair(d.post_text, 1)
            data = data.append(pair)
            pbar.update(1)
    data.columns = ['from_word', 'to_word', 'is_right']
    print(data.head())
    data.to_csv('./data/true_text.csv')

# %%
main()