import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import itertools
import tqdm


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def get_attr_vector(x, attr_to_id, pad = 0):
    v = [0]*(len(attr_to_id) + pad)
    #print(x)
    for a in x:
        #print(a)
        v[attr_to_id[a]] = 1
    return np.array(v)


TRAIN_RATIO = 0.7 # When training supervised model on single graph
NUM_PUNK_ATTR = 92
NUM_BID_ATTR = 3 + 22 + 14
GOLDEN_TYPES = ['Sold', 'Offered', 'Bid']
# Date deltas -- from fibannaci sequence
fib = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

# Price slices
cuts = [-1.] + [2 ** n / 10. for n in range(0,15)] + [10000.] + [19.2, 38.4, 76.8, 153.6, 307.2, 614.4]
cuts.sort()
ether_cuts = cuts
#eth_bins = list(zip(list(range(len(ether_cuts)-1)), ether_cuts[1:]))
# date_bin -- names
#date_bins = list(zip(list(range(len(fib)-1)), fib[1:]))
DATASET='2021-02-15'
#DATASET='2021-03-15'
#DATASET='2021-05-15'
PATH="./data/punks/"
def load_data(path=PATH, dataset=DATASET):
    print('Loading {} dataset...'.format(dataset))

    # Collect all nodes

    #################
    # Load punk nodes
    punk_df = pd.read_json(path+'punk_nodes.jsonl', lines=True)
    print(punk_df.shape, punk_df.keys())

    # Count 92 unique attributes
    attributes = sorted(set(itertools.chain.from_iterable(punk_df.attributes)))
    print(attributes, len(attributes))
    assert len(attributes) == NUM_PUNK_ATTR
    attr_to_id = {attr:i for i,attr in enumerate(attributes)}
    id_to_attr = {i:attr for i,attr in enumerate(attributes)}

    punk_df['attr_vector'] = punk_df.attributes.apply(lambda x: get_attr_vector(x, attr_to_id, pad = NUM_BID_ATTR))

    print(punk_df['attr_vector'])

    features = np.array(punk_df['attr_vector'])
    print(features.shape)
    print(features[1].shape)
    features = np.stack(features)
    print(features.shape)
    print(features[1:3,:])

    ###################
    # Load bid nodes
    bid_df = pd.read_json(path+'bid_nodes.jsonl', lines=True)
    print(bid_df.shape, bid_df.keys())

    # Nodes -- Bids ~ type/eth/date
    types = list(zip(list(range(3)), GOLDEN_TYPES))
    eth_bins = list(zip(list(range(len(ether_cuts)-1)), ether_cuts[1:]))
    date_bins = list(zip(list(range(len(fib)-1)), fib[1:]))
    print(types, '\n', eth_bins, '\n', date_bins)

    # Count 39 bid attributes
    bid_attr = [v for _, v in types] + [str(v) for _, v in eth_bins] + [str(v) for _, v in date_bins]
    print(bid_attr)
    assert len(bid_attr) == NUM_BID_ATTR
    attributes += bid_attr
    attr_to_id = {attr:i for i,attr in enumerate(attributes)}
    id_to_attr = {i:attr for i,attr in enumerate(attributes)}

    print(bid_df.head(n=11))

    bid_df['attr_vector'] = bid_df.apply(lambda x: get_attr_vector(x['desc'].split(';'), attr_to_id, pad = 0), axis=1)
    bid_features = np.stack(bid_df['attr_vector'])
    print(bid_features.shape)
    print(bid_features[1:4,:])

    # Merge features table...
    features = np.concatenate([features, bid_features], axis=0)
    features = torch.FloatTensor(features)
    print(features.shape)


    # Edges
    edges_df = pd.read_json(path+dataset+'_edges.jsonl', lines=True)
    print('edges', edges_df.shape, edges_df.keys())
    print(edges_df.sample(n=13))

    adj = np.zeros((features.shape[0], features.shape[0]))
    for _, edge_row in tqdm.tqdm(edges_df.iterrows(), total=len(edges_df)):
        pid = edge_row['punk_id']
        bid_desc = edge_row['bid_node_desc']
        bid_index = edge_row['bid_node_index']
        #print(pid, bid_desc, bid_index)
        adj[pid, bid_index] = 1
        adj[bid_index, pid] = 1

    # make sure matrix is square, and "normalized"
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])
    adj = torch.FloatTensor(adj)
    print(adj)


    # TODO: Add random edges... to avoid holes in the graph



    # Labels -- from prices
    prices_df = pd.read_json(path+dataset+'_prices.jsonl', lines=True)
    prices_df['sale_bucket'] = pd.cut(prices_df.sale, bins=ether_cuts, labels=range(len(ether_cuts)-1))
    prices_df['sale_bucket_max'] = pd.cut(prices_df.sale, bins=ether_cuts, labels=ether_cuts[1:])
    print('prices', prices_df.shape, prices_df.keys())

    # What are the high prices?
    print(prices_df[prices_df['sale_bucket'] >= 13])

    # Restrict to vertices... with a sale only -- really sparses
    sales = np.zeros((punk_df.shape[0]))
    for _,price_row in tqdm.tqdm(prices_df[~prices_df['sale'].isna()].iterrows(), total=prices_df[~prices_df['sale'].isna()].shape[0]):
        #print(price_row)
        pid = price_row['punk_id']
        sale = price_row['sale']
        sale_bucket = price_row['sale_bucket']
        sale_bucket_max = price_row['sale_bucket_max']
        #print(pid, sale, sale_bucket, sale_bucket_max)
        #sale_bucket = infer_sale_bucket(sale, buckets=ether_cuts)
        sales[pid] = sale_bucket
    print(sales)
    labels = torch.LongTensor(sales)
    print(torch.unique(labels, return_counts=True))


    # For now... select which vertices we want for train, val... and all the unknown we skip
    # All vertex that are not zero...
    print('nonzero', print(sales.shape))
    nonzero_sales = np.nonzero(sales.flatten())[0]
    #print(nonzero_sales)
    print(nonzero_sales.shape)
    # Numpy seeds!
    np.random.shuffle(nonzero_sales)
    train_split = int(len(nonzero_sales)*TRAIN_RATIO)
    idx_train = torch.LongTensor(nonzero_sales[:train_split])
    idx_val = torch.LongTensor(nonzero_sales[train_split:])
    idx_test = torch.LongTensor(nonzero_sales[train_split:])


    # What we got
    print('adj', adj.shape)
    print('features', features.shape)
    print('labels', labels.shape)
    print('idx_train', idx_train.shape)
    print('idx_val', idx_val.shape)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_old(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    print('features', features.shape)
    print('labels', labels.shape)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('adj, features, labels, idx_train, idx_val, idx_test')
    print(adj.shape, features.shape, labels.shape, idx_train.shape, idx_val.shape, idx_test.shape)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    print('normalizing matrix...')
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

