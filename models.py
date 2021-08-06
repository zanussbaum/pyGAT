import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_nodes=131, ndim_emb=16):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        print('Sparse attention')
        print(list(zip(['nfeat', 'nhid', 'nclass', 'dropout', 'alpha', 'nheads'], [nfeat, nhid, nclass, dropout, alpha, nheads])))


        # Learned embeddings for features
        self.emb = torch.nn.Embedding(n_nodes, embedding_dim=ndim_emb, scale_grad_by_freq=True)

        # Normalize the embeddings -- else can't roam free
        self.norm_emb = torch.nn.LayerNorm(ndim_emb, elementwise_affine=False)

        # Dropout between layers
        self.dropout = dropout

        # First layer of attentions
        self.attentions = [SpGraphAttentionLayer(ndim_emb, #nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.norm_att = torch.nn.LayerNorm(nhid*nheads, elementwise_affine=False)

        # Second layer of attentions
        self.attentions_two = [SpGraphAttentionLayer(nhid * nheads, #nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_two):
            self.add_module('attention_two_{}'.format(i), attention)
        self.norm_att_two = torch.nn.LayerNorm(nhid*nheads, elementwise_affine=False)

        # Third layer of attentions
        self.attentions_three = [SpGraphAttentionLayer(nhid * nheads, #nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_three):
            self.add_module('attention_three_{}'.format(i), attention)
        self.norm_att_three = torch.nn.LayerNorm(nhid*nheads, elementwise_affine=False)

        # Final attention layer...
        self.attentions_final = [SpGraphAttentionLayer(nhid * nheads,
                                             nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_final):
            self.add_module('attention_final_{}'.format(i), attention)


        self.norm_att_final = torch.nn.LayerNorm(nhid*nheads, elementwise_affine=False)

        # TODO: Could predict multiple things, like probability of sale, other aux
        self.out_head = torch.nn.Linear(nhid*nheads, nclass)

    def forward(self, x_in, adj, debug=False):
        if debug:
            print('------ forward -------')
            print(x_in.shape)

        # Embeddings... from onehots
        #print(x_in)
        non_zero = torch.nonzero(x_in)
        if debug:
            print('nonzero')
            print(torch.nonzero(x_in))

        # Just get all embeddings... and then do the matrix multiply
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        all_emb = self.emb(torch.arange(x_in.shape[1]).to(device))

        # Multiply the (sparse) onehot vector with embeddings.
        hot_emb = torch.mm(x_in, all_emb)
        x = hot_emb
        if debug:
            print(all_emb.shape)
            print(hot_emb.shape)
            print(hot_emb)
            print('----')

        # Do we need dropout on embedding layers?
        x = F.dropout(x, self.dropout, training=self.training)

        # Apply norm after embeddings -- else model blows up
        x = self.norm_emb(x)

        if debug:
            print(x)
            print(x.shape)

        # First round of attentions
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        if debug:
            print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # layer norm
        x = self.norm_att(x)

        if debug:
            print(x.shape)

        # Second round of attentions
        x = torch.cat([att(x, adj) for att in self.attentions_two], dim=1)
        if debug:
            print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # layer norm
        x = self.norm_att_two(x)

        if debug:
            print(x.shape)

        """
        # Third round of attentions
        x = torch.cat([att(x, adj) for att in self.attentions_three], dim=1)
        if debug:
            print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # layer norm
        x = self.norm_att_three(x)

        if debug:
            print(x.shape)
        """

        # Final round of attention(s)
        #x = self.out_att(x, adj)
        x = torch.cat([att(x, adj) for att in self.attentions_final], dim=1)

        # Skip dropout before final layer?
        x = F.dropout(x, self.dropout, training=self.training)
        # layer norm
        x = self.norm_att_final(x)

        # Non-linearity?
        x = F.elu(x)
        if debug:
            print(x.shape)

        # TODO: Insert small MLP to final embedding dimension.
        out = F.log_softmax(self.out_head(x), dim=1)
        if debug:
            print(out.shape)
        return out

