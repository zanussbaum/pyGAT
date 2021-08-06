import torch 
import argparse
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from utils import load_df, DATASET
from tqdm import tqdm




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

    
def get_punk_similars(punk_id, epoch, verbose=False, n=10):
    attention = torch.load(f'embeddings/attention_{DATASET}_{epoch}.pt').detach().cpu().numpy()
    print(attention.shape)


    similarities = get_similarity(attention)
    punk_df, bid_df, edges_df = load_df()

    total_df = pd.concat([punk_df, bid_df], axis=0)

    pairs = get_most_similar(similarities, punk_id)

    if verbose: 
        print(f"Ref Punk: {punk_id}, Type: {total_df['type'].iloc[punk_id]}, Attributes: {total_df['attributes'].iloc[punk_id]}\n")

        for pair in pairs:
            index = pair['index']
            _, close_punk = index[0], index[1]
            rounded_score = str(round(pair['score'], 2))
            print(f"PunkId: {close_punk}, Score: {rounded_score} Type: {total_df['type'].iloc[close_punk]}, Attributes: {total_df['attributes'].iloc[close_punk]}")

    return pairs

        
def get_similarity_embeddings(epoch, verbose=False, n=10):
    punk_df, bid_df, edges_df = load_df()
    total_df = pd.concat([punk_df, bid_df], axis=0) 

    dfs = []
    for punk_id in tqdm(range(len(punk_df))):
        dfs.append(pd.DataFrame(get_punk_similars(punk_id, epoch, verbose=verbose, n=n)))

        
    return pd.concat(dfs, axis=0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=42, help='Checkpoint model to load')
    parser.add_argument('--date', type=str, help='Checkpoint model to load')
    parser.add_argument('--punk_id', type=int, default=0, help='Punk id to use')


    args = parser.parse_args()

    return args

    
if __name__ == '__main__':
    args = parse_args()
    if args.punk_id:
       get_punk_similars(args.punk_id, args.epoch, verbose=True, n=10) 
    else:
        df = get_similarity_embeddings(verbose=False, n=10)
        df.to_csv(f'similarity_embeddings_{args.date}.csv', index=False)

