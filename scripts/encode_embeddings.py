from sentence_transformers import SentenceTransformer as embedder
import torch
from similearn.wandb_ import WandbClient
import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm

multiple_gpus = False

# load model
print("Loading the model...")
model_name = "BAAI/llm-embedder"
encoder = embedder(model_name)


# load datasets from wandb
run_name = model_name.split("/")[-1] + "-evaluation"
wandbc = WandbClient(run_name=run_name)


# load datasets from wandb
corpus = pd.read_parquet(wandbc.load_dataset("retrieval_corpus"))
corpus.reset_index(inplace=True, drop=True)
gold = pd.read_parquet(wandbc.load_dataset("gold_dataset"))


# encode corpus
# pre GPUs
if multiple_gpus:
    gpu1 = torch.device("cuda:0")
    gpu2 = torch.device("cuda:1")

    pool = encoder.start_multi_process_pool()

    encoded_corpus = encoder.encode_multi_process(
        corpus["text"], pool=pool, batch_size=64
    )

print("Encoding the corpus...")
encoded_corpus = encoder.encode(
    corpus["text"], batch_size=32, show_progress_bar=True
)

# index corpus
print("Indexing the corpus...")
vector_dimension = encoded_corpus.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(encoded_corpus)
index.add(encoded_corpus)


def compute_f1(gold, ann):
    P = len(set(gold).intersection(set(ann))) / 10
    R = len(set(gold).intersection(set(ann))) / len(gold)
    F1 = 2 * P * R / (P + R)
    return F1


def get_list_of_ids(ann, lookup):
    anns_ids = []
    for idx in ann:
        anns_ids.append(lookup.loc[idx])

    return anns_ids

print("Evaluating...")
id_lookup = corpus["id"]
f1s = []
for idx, row in tqdm(gold.iterrows()):
    search_vector = encoder.encode(row["query"])
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    k = 10
    distances, ann = index.search(_vector, k=k)
    ann = get_list_of_ids(ann, id_lookup)
    f1s.append(compute_f1(row["recs"], ann))

print(np.mean(f1s))
