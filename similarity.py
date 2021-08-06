import torch 
import argparse
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from utils import load_df
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, help='Checkpoint model to load')
parser.add_argument('--date', type=str, help='Checkpoint model to load')
parser.add_argument('--punk_id', type=int, default=0, help='Punk id to use')


args = parser.parse_args()

def get_similarity(emb):
    print("Running similarity")
    similarities = cosine_similarity(emb)

    return similarities

def get_most_similar(similarities, punk_id, n=10):
    punk_similarities = similarities[punk_id]

    pairs = []
    for i in range(len(similarities)):
        if i == punk_id:
            continue
        pairs.append({'index': [punk_id, i], 'score': similarities[punk_id][i]}) 
    
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)[:n]

    return pairs


punk_id = args.punk_id 

attention = torch.load(f'embeddings/attention_{args.epoch}.pt')
print(attention.shape)


similarities = get_similarity(attention)
punk_df, bid_df, edges_df = load_df()

total_df = pd.concat([punk_df, bid_df], axis=0)

pairs = get_most_similar(similarities, punk_id)

print(f"Ref Punk: {punk_id}, Type: {total_df['type'].iloc[punk_id]}, Attributes: {total_df['attributes'].iloc[punk_id]}\n")

for pair in pairs:
    index = pair['index']
    _, close_punk = index[0], index[1]
    rounded_score = str(round(pair['score'], 2))
    print(f"PunkId: {close_punk}, Score: {rounded_score} Type: {total_df['type'].iloc[close_punk]}, Attributes: {total_df['attributes'].iloc[close_punk]}")
