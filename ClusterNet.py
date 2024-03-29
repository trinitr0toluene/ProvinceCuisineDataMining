from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import openpyxl
import heapq
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset
import scipy.sparse as sp
from models import cluster, GCNClusterNet, GCN
from torch_geometric.utils.convert import to_networkx
from sklearn.manifold import TSNE
from log import get_logger
import datetime
from CalModularity import Q
import torch_geometric
import os
import json
# from pylab import mpl
 
# # 设置中文显示字体
# mpl.rcParams["font.sans-serif"] = ["Noto Mono"]
# # 设置正常显示符号
# mpl.rcParams["axes.unicode_minus"] = False



# from torch.utils.tensorboard import SummaryWriter
def parser_args():
    parser = argparse.ArgumentParser(description='ClusterNet Training')


    parser.add_argument('--epochs', default=1001, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--label', default = 0, type = int, help = 'label pattern for node class')
    parser.add_argument('--num_class', default = 3, type = int, help = 'number of class for clustering')
    parser.add_argument('--num_edge', default = 3, type = int, help = 'number of edge: num_province*num_edge')
    parser.add_argument('--nhid', default = 50, type = int, help = 'hidden dim of ClusterNet')
    parser.add_argument('--nout', default = 50, type = int, help = 'output dim of ClusterNet')
    parser.add_argument('--dropout', default = 0.2, type = float, help = 'dropout parameter')
    parser.add_argument('--cluster_temp', default = 50, type = int, help = 'cluster_temp for ClusterNet')
    parser.add_argument('--wd', default = 5e-4, type = float, help = 'weight decay')

    # parser.add_argument('--output', default = '/home/zhangziyi/code/ProvinceCuisineDataMining/Config')
    
    
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args

args = get_args()

edge_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/edge_features.xlsx'
node_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_features.xlsx'
label_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_class.xlsx'

K = args.num_class

start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = get_logger(start_time)
# logger.get_logger()
# logger.add_handler(start_time)
logger.info("Begin")
logger.info(f'label:{args.label}')
logger.info(f'社区数目K：{args.num_class}')

config_path = '/home/zhangziyi/code/ProvinceCuisineDataMining/Log/'+start_time[:10]+'/'+start_time[11:]+'/_config.json'

with open(config_path, 'wt') as f:
        json.dump(vars(args), f, indent=4) 
logger.info("Full config saved to {}".format(config_path))



if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



#initialize
edge_index = []
start = []
end = []

edge_attr = []
node_feature = []
# label=torch.rand(31)
label = []
province_list = []

def excel2edge(): 
    # open the edge features file and transform it into PyG version
    # param: filename
    # return: edge_index, edge_attr

    df = pd.read_excel(f'{edge_filepath}', engine='openpyxl', sheet_name=0) 
    # 
    
    # print(df.columns.values)
    # print(df.index)
    # print(df.shape)
    rows = df.shape[0]
    columns = df.shape[1]

    # print(rows)
    # print(columns)
    

    # for columns

    for i in range(columns):
        data_list = df.iloc[:,i].tolist()
        # print(i)
        for j in range(rows):
            data = df.iloc[j,i]
            if(isTopK(data, data_list)):
                start.append(j)
                end.append(i)
                global edge_attr
                edge_attr.append(data)
                # print(str(j) + ',' + str(i) + ',' + str(data))
            
            else:
                # print('第'+j+'行'+i+'列的数据为'+data+',非Top'+k+',故舍弃该边')
                # print(str(df.columns.values[i]) + '到' + str(df.columns.values[j]) + '人口流动数据为' + str(data) + ',非top' + str(k) + ',舍弃该边')
                continue
    
    global edge_index
    
    edge_index.append(start)
    edge_index.append(end)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    print(edge_attr.size())
    print(edge_index.size())
    return edge_index, edge_attr


def excel2node():
    df = pd.read_excel(f'{node_filepath}', engine='openpyxl', sheet_name=0)  

    
    # print(df.shape)
    rows = df.shape[0]
    columns = df.shape[1]

    # print(rows)
    # print(columns)

    df_new = df.drop(df.columns[[0,1]], axis=1)
    
    # print(df_new.shape)
    global node_feature
    node_feature = torch.tensor(df_new.values, dtype=torch.float)

    # print(node_feature)
    # print(node_feature.size())

    return node_feature

 

def isTopK(data, data_list):

    max_val_lis = heapq.nlargest(args.num_edge, data_list)
    # print(max_val_lis)
    
    if(data in max_val_lis):
        return True
    else:
        return False


def addClass():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=args.label)
    df_new = df.drop(df.columns[[0,1]], axis=1)
    df_new['max_idx'] = df_new.idxmax(axis=1)
    # print(df_new['max_idx'])
    global label
    
    label = torch.tensor(df_new['max_idx'], dtype=torch.int)
     
    
    # print(label)
    # print(label.size())
    return label


def getProvince():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=0)
    global province_list
    province_list = df.values[: , 0]
    return province_list
   
# build the PyG Dataset




"""
        Args:
            x (Tensor, optional): 节点属性矩阵，大小为`[num_nodes, num_node_features]`
            edge_index (LongTensor, optional): 边索引矩阵，大小为`[2, num_edges]`，第0行为尾节点，第1行为头节点，头指向尾
            edge_attr (Tensor, optional): 边属性矩阵，大小为`[num_edges, num_edge_features]`
            y (Tensor, optional): 节点或图的标签，任意大小（，其实也可以是边的标签）
"""

 
#
class MyOwnDataset(InMemoryDataset):
    def __init__(self, 
                 root, 
                 transform=None, 
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return []
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return 'data.pt'
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        node_feature=excel2node()
        edge_index, edge_attr=excel2edge()
        label = addClass()
        
        data = Data(x=node_feature, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    y=label)
        # print(label)
        # print(data.y)
        data_list = [data]
        
 
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
 
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = MyOwnDataset(root='/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/label'+str(args.label)+'/')

# print(dataset.num_classes) # 0
# print(dataset[0].num_nodes) # 31
# print(dataset[0].num_edges) # 93
# print(dataset[0].num_features) # 8

data_list = [dataset[0]]
data = dataset[0]
G = to_networkx(data)
# print(data)
# print(type(data))

# transform = RandomLinkSplit(is_undirected=True)
# print(type(data))
# data = transform(data)
# data = train_test_split_edges(data)
# print(data)
# print(type(data))
# print(data.x)
# loader = DataLoader(data_list, batch_size=1)



class GCN_NET(torch.nn.Module):
    def __init__(self, nhid, nout, dropout):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, nhid)
        self.conv2 = GCNConv(nhid,nout)
        self.dropout = dropout
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,self.dropout , training=self.training)
        x = self.conv2(x, edge_index)
        return x

"""Row-normalize sparse matrix"""
def normalize(mx):
	
    rowsum = np.array(mx.sum(1))  # 矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 每行和的-1次方
    r_inv[np.isinf(r_inv)] = 0.  # 如果是inf，转换为0
    r_mat_inv = sp.diags(r_inv)  # 转换为对角阵
    mx = r_mat_inv.dot(mx)  # D-1*A,乘上特征，按行归一化
    return mx

"""Convert a scipy sparse matrix to a torch sparse tensor."""
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GCNClusterNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, K, cluster_temp):
        super(GCNClusterNet, self).__init__()
        self.GCN = GCN(nfeat, nhid, nout, dropout)
        self.distmult = torch.nn.Parameter(torch.rand(nout))
        self.sigmoid = torch.nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init = torch.rand(self.K, nout)

    def forward(self, x,adj, num_iter=1):
        
        embeds = self.GCN(x, adj)
        mu_init, _, _ = cluster(embeds, self.K, num_iter, cluster_temp=self.cluster_temp, init=self.init)
        mu, r, dist = cluster(embeds, self.K, 1, cluster_temp=self.cluster_temp, init=mu_init.detach().clone())
        return mu, r, embeds, dist





def make_modularity_matrix(adj):
    adj = adj * (torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(axis=0).unsqueeze(1)
    mod = adj - degrees @ degrees.t() / adj.sum()
    return mod

def array2list(com):
    temp = []
        
    for j in range(K):
        l = []
        temp.append(l)
        for i in range(G.number_of_nodes()):
            if com[i] == j:               
                temp[j].append(i)

    print(temp)
    com = temp
    return com
    


def loss_modularity(r, bin_adj, mod):
    bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()

def draw_nx(com):
    temp = array2list(com)
    com = temp
    
    pos = nx.spring_layout(G) # 节点的布局为spring型
    NodeId    = list(G.nodes())
    # print(f'NodeId:{NodeId}')
    # logger.info(f'NodeId:{NodeId}')
    node_size = [G.degree(i)**1.2*90 for i in NodeId] # 节点大小

    plt.figure(figsize = (8,8)) # 设置图片大小
    # print(pos)
    # print(type(com[1]))
    nx.draw(G,pos, 
            with_labels=True, 
            node_size =node_size, 
            node_color='w', 
            node_shape = '.'
        )

    '''
    node_size表示节点大小
    node_color表示节点颜色
    with_labels=True表示节点是否带标签
    '''
    color_list = ['pink','orange','r','g','b','y','m','gray','c','brown', '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
                '#ffd700']
    # print(len(com))
    # print(f'type of com:{type(com)}')
    # print(f'type of pos:{type(pos)}')
    for i in range(K):
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=com[i], 
                            node_color = color_list[i+2],  
                            label=True)
    plt.show()
    plt.savefig('/home/zhangziyi/code/ProvinceCuisineDataMining/Log/'+start_time[:10]+'/'+start_time[11:]+'/ClusterNet_nx')

def draw(z,r):
    colors = [
            'pink','orange','r','g','b','y','m','gray','c','brown', '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
            '#ffd700']
    
    # 使用TSNE先进行数据降维，形状为[num_nodes, 2]
    z = TSNE(n_components=2).fit_transform(z.detach().numpy())
    getProvince()
    # z = np.c_[z,province_list]
    
    # z.append(province_list)
    # print(z)

    
    plt.figure(figsize=(8, 8))
    
    result = r
    for j in range(K):
        plt.scatter(z[result == j,0], z[result == j,1],s=450, color=colors[j], alpha=0.5)
    for i in range(z.shape[0]):  ## for every node
        plt.annotate(province_list[i], xy = (z[i,0],z[i,1]),  xytext=(-20, 10), textcoords = 'offset points',ha = 'center', va = 'top')    

    # for i in range(z.shape[0]):  ## for every node
    #     for j in range(K):
    #         plt.scatter(z[i,0], z[i,1],s=450, color=colors[j], alpha=0.5)
    #         print(z[i,0])
    #         print(z[i,1])
    #         plt.annotate(province_list[i], xy = (z[i,0],z[i,1]), textcoords = 'offset points',ha = 'center', va = 'top')

    # 绘制不同类别的节点
    # for i in range(dataset.num_classes):
    #     # z[y==0, 0] 和 z[y==0, 1] 分别代表第一个类的节点的x轴和y轴的坐标
    #     plt.scatter(z[y == i, 0], z[y == i, 1], s=450, color=colors[i], alpha=0.5)
        
        # plt.annotate(z[2,] xy = (x,y), textcoords = 'offset points',ha = 'center', va = 'top')
    plt.axis('off')
    plt.show()
    plt.savefig('/home/zhangziyi/code/ProvinceCuisineDataMining/Log/'+start_time[:10]+'/'+start_time[11:]+'/cluster')    



features = sp.csr_matrix(data.x, dtype=np.float32)
# print(data.y)
  # 取特征
# adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0, :], 			data.edge_index[1, :])),
#                     shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0, :], 			data.edge_index[1, :])),
                    shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
features = normalize(features)  # 特征归一化
adj = normalize(adj + sp.eye(adj.shape[0]))  # A+I归一化
features = torch.FloatTensor(np.array(features.todense()))# 将numpy的数据转换成torch格式
adj = sparse_mx_to_torch_sparse_tensor(adj)
adj = adj.coalesce()
bin_adj_all = (adj.to_dense() > 0).float()
print(f'adj:{adj} bin_adj_all:{bin_adj_all}')

'''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the
    embeddings and the the node similarities (just output for debugging purposes).

    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to
    run the k-means updates for.
    '''





test_object = make_modularity_matrix(bin_adj_all)

num_cluster_iter = 1
losses = []


model_cluster = GCNClusterNet(nfeat=data.x.size(1), nhid=args.nhid, nout=args.nout, dropout=args.dropout, K=K, cluster_temp=args.cluster_temp)
optimizer = torch.optim.Adam(model_cluster.parameters(), lr=args.lr, weight_decay=args.wd)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cluster.train()

'''
    torch.Size([31, 50])
    tensor([[ 0.1287,  0.1506,  0.0398,  ..., -0.0872, -0.0940,  0.0876],
        [ 0.1315,  0.1490,  0.0389,  ..., -0.0874, -0.0979,  0.0880],
        [ 0.1303,  0.1499,  0.0411,  ..., -0.0853, -0.0955,  0.0878],
        ...,
        [ 0.1330,  0.1531,  0.0369,  ..., -0.0863, -0.0991,  0.0893],
        [ 0.1325,  0.1513,  0.0369,  ..., -0.0847, -0.0952,  0.0905],
        [ 0.1305,  0.1516,  0.0384,  ..., -0.0873, -0.0952,  0.0897]],
       grad_fn=<AddBackward0>)
    '''

iter_num = args.epochs
best_test_loss = 0
for epoch in range(iter_num):
    
    
    mu, r, embeds, dist = model_cluster(features, adj, num_cluster_iter)
    # print(embeds.size())
    # print(embeds)

    
    loss = loss_modularity(r, bin_adj_all, test_object)
    loss = -loss
    optimizer.zero_grad()
    loss.backward()
    if epoch == 500:
        num_cluster_iter = 5
    if epoch % 100 == 0:
        r = torch.softmax(100 * r, dim=1)
        # if epoch ==1000:
        #     print(f"前10行训练得到分配矩阵{r[0:10,0:K]}")
            # print(r)
            # colors = [
            #  '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
            #  '#ffd700','green']
            # z = embeds
            # # 使用TSNE先进行数据降维，形状为[num_nodes, 2]
            # z = TSNE(n_components=2).fit_transform(z.detach().numpy())
            # y = data.y.detach().numpy()

            # plt.figure(figsize=(8, 8))
            
            # # 绘制不同类别的节点
            # for i in range(dataset.num_classes):
            #     # z[y==0, 0] 和 z[y==0, 1] 分别代表第一个类的节点的x轴和y轴的坐标
            #     plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
            # plt.axis('off')
            # plt.show()
            # plt.savefig('./dataset')
            

    loss_test = loss_modularity(r, bin_adj_all, test_object)
    if epoch == 0:
        best_train_val = 100
    if loss.item() < best_train_val:
        best_train_val = loss.item()
        curr_test_loss = loss_test.item()
        # convert distances into a feasible (fractional x)#将距离转换为可行的（分数x）
        x_best = torch.softmax(dist * 100, 0).sum(dim=1)
        x_best = 2 * (torch.sigmoid(4 * x_best) - 0.5)
        if x_best.sum() > 5:
            x_best = 5 * x_best / x_best.sum()
    losses.append(loss.item())
    optimizer.step()
    
    if epoch == iter_num-1:
        logger.info(f'r:{r}')
        r = r.detach().numpy()
        r = np.argmax(r, axis=1)

        # print(type(r))
        adj_array = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index)
        adj_array = adj_array.toarray()
        # print(adj_array)
        #计算模块度

        score = Q(adj_array, r)
        print(f'模块度为：{score}')
        logger.info(f'模块度为：{score}')
        draw(embeds, r)
        draw_nx(r)
        com = array2list(r)
        print(f'nx库计算结果：{nx.community.modularity(G, com)}')
        logger.info(f'nx库计算结果：{nx.community.modularity(G, com)}')

 

    logger.info(f'epoch{epoch + 1}   ClusterNet value:{curr_test_loss}')
    print(f'epoch{epoch + 1}   ClusterNet value:{curr_test_loss}')
    if curr_test_loss > best_test_loss:
        best_test_loss = curr_test_loss
        es = 0
    else:
        es += 1
        if es == 200:
            
            print('Early Stop!')
            logger.info('Early Stop!')
            r = r.detach().numpy()
            r = np.argmax(r, axis=1)

            print(type(r))
            print(r)
            adj_array = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index)
            adj_array = adj_array.toarray()
            # print(adj_array)
            #计算模块度
            score = Q(adj_array, r)
            print(f'模块度为：{score}')
            logger.info(f'模块度为：{score}')
            draw(embeds, r)
            draw_nx(r)
            com = array2list(r)
            print(f'nx库计算结果：{nx.community.modularity(G, com)}')
            logger.info(f'nx库计算结果：{nx.community.modularity(G, com)}')

            break




    
