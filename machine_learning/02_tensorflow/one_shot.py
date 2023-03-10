import string

import numpy as np

if __name__ == '__main__':
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                # 为每个唯一单词指定一个唯一索引
                token_index[word] = len(token_index) + 1

    print(token_index)

    # 对样本进行分词，只考虑每个样本前max_length个单词
    max_length = 10
    result = np.zeros(shape=(len(samples),
                             max_length,
                             max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            result[i, j, index] = 1

    print(result)

    # one-hot散列技巧
