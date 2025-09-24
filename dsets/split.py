from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import SpectralClustering
from pathlib import Path
import json
import random

def split(data_dir:str, ds_name:str, num_edits:int = 10000, num_clients:int = 5, num_time_slots:int = 10, batch_size = 128, rate = 0.7):
    # load bert model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').cuda()
    model.eval()

    # load data
    data_dir = Path(data_dir)
    if ds_name == "zsre":
        zsre_loc = data_dir / "zsre_mend_eval.json"
        with open(zsre_loc, "r") as f:
            raw = json.load(f)
        sentences = [entry['src'] for entry in raw[:num_edits]]
    elif ds_name == "mcf":
        mcf_loc = data_dir / "multi_counterfact.json"
        with open(mcf_loc, "r") as f:
            raw = json.load(f)
        sentences = [entry['requested_rewrite']['prompt'].format(entry['requested_rewrite']['subject']) for entry in raw[:num_edits]]

    embeddings = []
    for i in range(0, num_edits, batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.cuda() for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu().numpy())
    embeddings = np.vstack(embeddings)

    similarity_matrix = cosine_similarity(embeddings)

    spectral_cluster = SpectralClustering(n_clusters=num_clients, affinity='precomputed',random_state=0)
    cluster_labels = spectral_cluster.fit_predict(similarity_matrix)

    cluster_dict = {}
    for index, label in enumerate(cluster_labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(index)
    
    ids_pool = []
    num_per = len(cluster_labels) // num_clients
    for label, case_ids in cluster_dict.items():
        num_keep = int(rate*len(case_ids))
        cluster_dict[label] = case_ids[:num_keep]
        ids_pool.extend(case_ids[num_keep:])
    random.seed(0)
    random.shuffle(ids_pool)

    index = 0

    for label in range(num_clients):
        delta = num_per - len(cluster_dict[label])
        cluster_dict[label].extend(ids_pool[index:index+delta])
        index = index + delta

    client_index = {}
    for client in range(num_clients):
        client_index[client] = {}
        n = num_per // num_time_slots
        for t in range(num_time_slots):
            client_index[client][t] = cluster_dict[client][t*n:t*n+n]

    return client_index


if __name__ == '__main__':
    client_case_index = split('data', ds_name='mcf', num_time_slots=1)
    print(client_case_index)
    def count_values(d):
        count = 0
        for value in d.values():
            if isinstance(value, dict):
                count += count_values(value)  # 递归调用，统计嵌套字典中的值
            else:
                count += len(value) # 统计当前层级的值
        return count

    # 统计字典中所有值的数量
    total_values = count_values(client_case_index)
    print(f"字典中所有值的数量是：{total_values}")


    
         