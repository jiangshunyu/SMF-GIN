import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp import MLP

class GraphCNN(nn.Module):
    def __init__(self, args, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighboor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.weight = nn.Parameter(torch.ones(self.num_layers - 1))
        #self.sub_weight = nn.Parameter(torch.zeros(1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.type = args.type
        self.attention_type = args.attention_type

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        # #attention by chenweijian1
        if self.attention_type == 'mlp':
            self.mlp_layer = 2
            self.mlp_layer_stack = nn.ModuleList()
            for i in range(self.mlp_layer):
                if i < self.mlp_layer - 1:
                    self.mlp_layer_stack.append(
                        nn.Linear(hidden_dim * 4, hidden_dim * 4, bias=True))
                else:
                    self.mlp_layer_stack.append(
                        nn.Linear(hidden_dim * 4, hidden_dim, bias=True))
        elif self.attention_type == 'attention':
            self.w = nn.Parameter(torch.Tensor(hidden_dim * 4, hidden_dim))
            self.bias = nn.Parameter(torch.Tensor(hidden_dim))
            self.attn = nn.Parameter(torch.Tensor(hidden_dim, 1))

            self.softmax = nn.Softmax(dim=-1)

            nn.init.xavier_uniform_(self.w)
            nn.init.constant_(self.bias, 0)
            nn.init.xavier_uniform_(self.attn)
        elif self.attention_type == 'self-attention':
            self.selfatt = nn.MultiheadAttention(hidden_dim * 4, 2)
        elif self.attention_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim*4, nhead=1)
            self.trans = nn.TransformerEncoder(encoder_layer, num_layers=2)


        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))


    def attention_local(self, x1, x2):

        if self.attention_type == 'weight':
            return 0.7*x1 + 0.3*x2
        elif self.attention_type == 'mlp':
            user_feature = torch.stack((x1, x2), 1)
            for i, mlp_layer in enumerate(self.mlp_layer_stack):
                user_feature = mlp_layer(user_feature)
                if i + 1 < self.mlp_layer:
                    user_feature = F.relu(user_feature)
            user_feature = torch.max(user_feature, dim=1)[0]
            return user_feature
        elif self.attention_type == 'attention':
            item_feature = torch.stack((x1, x2), 1).transpose(0,1)
            item_feature = torch.matmul(item_feature, self.w) + self.bias
            attn_coef = torch.matmul(torch.tanh(item_feature), self.attn)
            attn_coef = self.softmax(attn_coef).transpose(1,2)
            x = torch.matmul(attn_coef, item_feature).transpose(0,1)
            x = x.squeeze(1)
            print(x.shape)
            return x
        elif self.attention_type == 'self-attention':
            out = torch.stack((x1, x2), 1).transpose(0,1)
            out = self.selfatt(out, out, out)[0].transpose(0,1)

            #pooling strategies
            out = torch.max(out, dim=1)[0]
            # out = torch.mean(out, dim = 1)
            # out = torch.sum(out, dim = 1)
            return out
        elif self.attention_type == 'transformer':
            out = torch.stack((x1, x2), 1).transpose(0, 1)
            out = self.trans(out).transpose(0, 1)

            #pooling strategies
            out = torch.max(out, dim=1)[0]
            # out = torch.mean(out, dim = 1)
            # out = out[0]
            return out



    def attention_global(self, x1,x2,x3,x4):
        if self.attention_type == 'weight':
            return self.weight[0]*x1 + self.weight[1]*x2 + self.weight[2]*x3 + self.weight[3]*x4
        elif self.attention_type == 'mlp':
            user_feature = torch.stack((x1, x2, x3, x4), 1)
            for i, mlp_layer in enumerate(self.mlp_layer_stack):
                user_feature = mlp_layer(user_feature)
                if i + 1 < self.mlp_layer:
                    user_feature = F.relu(user_feature)
            user_feature = torch.max(user_feature, dim=1)[0]
            return user_feature
        elif self.attention_type == 'attention':
            item_feature = torch.stack((x1, x2, x3, x4), 1).transpose(0, 1)
            item_feature = torch.matmul(item_feature, self.w) + self.bias
            attn_coef = torch.matmul(torch.tanh(item_feature), self.attn)
            attn_coef = self.softmax(attn_coef).transpose(1, 2)
            x = torch.matmul(attn_coef, item_feature).transpose(0, 1)
            x = x.squeeze(1)
            print(x.shape)
            return x
        elif self.attention_type == 'self-attention':
            out = torch.stack((x1, x2, x3, x4), 1).transpose(0, 1)
            out = self.selfatt(out, out, out)[0].transpose(0, 1)

            # pooling strategies
            out = torch.max(out, dim=1)[0]
            # out = torch.mean(out, dim = 1)
            # out = torch.sum(out, dim = 1)
            return out
        elif self.attention_type == 'transformer':
            out = torch.stack((x1, x2, x3, x4), 1).transpose(0, 1)
            out = self.trans(out).transpose(0, 1)

            # pooling strategies
            out = torch.max(out, dim=1)[0]
            # out = torch.mean(out, dim = 1)
            # out = out[0]
            return out


    def __preprocess_neighbors_maxpool(self, batch_graph):



        max_deg = max([graph.max_neighbors for graph in batch_graph])

        padded_neighbors_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):

                pad = [n + start_idx[i] for n in graph.neighbors[j]]

                pad.extend([-1]*(max_deg - len(pad)))


                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbors_list.extend(padded_neighbors)

        return torch.LongStorage(padded_neighbors_list)


    def __preprocess_neighbors_sumavepool(self, batch_graph):


        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list,1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])



        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node),range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)


    def __preprocess_graphpool(self, batch_graph):


        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):

            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))

            else:
                elem.extend([1]*len(graph.g))


            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph),start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbors_list):


        dummy = toch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1,-1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbors_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbors_list = None, Adj_block = None):


        if self.neighboor_pooling_type == "max":

            pooled = self.maxpool(h, padded_neighbors_list)
        else:

            pooled = torch.spmm(Adj_block, h)
            if self.neighboor_pooling_type == "average":

                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0],1)).to(self.device))
                pooled = pooled/degree


        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)


        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbors_list = None, Adj_block = None):


        if self.neighboor_pooling_type == "max":

            pooled = self.maxpool(h, padded_neighbors_list)
        else:

            pooled = torch.spmm(Adj_block, h)
            if self.neighboor_pooling_type == "average":

                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0],1)).to(self.device))
                pooled = pooled/degree


        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)


        h = F.relu(h)
        return  h


    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighboor_pooling_type == "max":
            padded_neighbors_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)


        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers-1):
            if self.neighboor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbors_list = padded_neighbors_list)
            elif not self.neighboor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighboor_pooling_type =="max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbors_list = padded_neighbors_list)
            elif not self.neighboor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)

            hidden_rep.append(h)


        s = []

        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            pooled_h = F.dropout(pooled_h, self.final_dropout, training=self.training)
            s.append(pooled_h)

        if self.type == 'local':
            out = torch.cat((s[1], s[2], s[3], s[4]), 1)
            out1 = torch.FloatTensor(int(out.size(0) / 2), out.size(1)).to(self.device)
            out2 = torch.FloatTensor(int(out.size(0) / 2), out.size(1)).to(self.device)
            return out1, out2
        elif self.type == 'global':
            return s[1], s[2], s[3], s[4]

