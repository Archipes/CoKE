from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')
with open('zsre_mend_eval.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

src = [entry['src'] for entry in data[:200]]

embeddings = model.encode(src)
similarity_matrix = cosine_similarity(embeddings)
similarity_matrix = (similarity_matrix + 1) / 2

similarities = []
for i in range(len(src)):
    for j in range(i + 1, len(src)):
        similarities.append(similarity_matrix[i][j])


folder_path = "EleutherAI_gpt-j-6B_MEMIT"

# 获取文件夹中所有 .npz 文件的完整路径
file_list = ([os.path.join(folder_path, f"zsre_layer_8_clamp_0.75_case_{index}.npz") for  index in range(200)])
# 读取所有文件到一个列表中
zs_list = []
for file_path in file_list:
    zs_ = np.load(file_path)['v_star']  # 加载 .npz 文件
    zs_list.append(zs_)     # 将加载的内容存入列表中

similarity_matrix_zs = cosine_similarity(zs_list)
similarity_matrix_zs = (similarity_matrix_zs + 1) / 2

similarities_zs = []
for i in range(len(zs_list)):
    for j in range(i + 1, len(zs_list)):
        similarities_zs.append(similarity_matrix_zs[i][j])



plt.figure(figsize=(10, 6))
plt.plot(range(1, len(similarities) + 1), similarities, linestyle='-', color='b', label="Curve 1")
plt.plot(range(1, len(similarities_zs) + 1), similarities_zs, linestyle='-', color='r', label="Curve 2")
plt.title("Cosine Similarity Between Sentence Pairs")
plt.xlabel("Sentence Pair Index")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.show()