# dataの組み合わせによるモデルの精度を比較するためのプログラム


import itertools

# これはあなたのdata_partsリストです
data_parts = ["part1", "part2", "part3", "part4"]

# すべての可能な組み合わせを生成します
for r in range(1, len(data_parts) + 1):
    for combination in itertools.combinations(data_parts, r):
        print(combination)