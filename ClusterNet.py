from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import openpyxl
import heapq

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset
import scipy.sparse as sp
from models import cluster, GCNClusterNet, GCN
from sklearn.manifold import TSNE
from log import get_logger
import datetime
# from pylab import mpl
 
# # 设置中文显示字体
# mpl.rcParams["font.sans-serif"] = ["Noto Mono"]
# # 设置正常显示符号
# mpl.rcParams["axes.unicode_minus"] = False



# from torch.utils.tensorboard import SummaryWriter


edge_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/edge_features.xlsx'
node_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_features.xlsx'
label_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_class.xlsx'
k = 3

K = 3
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

    max_val_lis = heapq.nlargest(k, data_list)
    # print(max_val_lis)
    
    if(data in max_val_lis):
        return True
    else:
        return False


def addClass():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=0)
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

dataset = MyOwnDataset(root='/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/')

# print(dataset.num_classes) # 0
# print(dataset[0].num_nodes) # 31
# print(dataset[0].num_edges) # 93
# print(dataset[0].num_features) # 8

data_list = [dataset[0]]
data = dataset[0]
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

# #  定义2层GCN的网络.
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)
    
    
#     def forward(self):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr  #赋值data.x特征向量edge_index图的形状，edge_attr权重矩阵

#         x = self.conv1(x, edge_index, edge_weight)
#         x = F.relu(x)   
#         #x,edge_index,edge_weight特征矩阵，邻接矩阵，权重矩阵组成GCN核心公式
#         x = F.dropout(x, training=self.training)   #用dropout函数防止过拟合
#         x = self.conv2(x, edge_index, edge_weight)  #输出
#         print(x)
#         return x
#         #x为节点的embedding

 
    
# # 5.3) 训练 & 测试.

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# # 训练模型
# def train():
#     model.train()#设置成train模式
#     optimizer.zero_grad()#清空所有被优化的变量的梯度
#     F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward() #损失函数训练参数用于节点分类
#     optimizer.step()#步长
     
# @torch.no_grad()#不需要计算梯度，也不进行反向传播


# # for epoch in range(200):
# #     loss_all = 0
# #     for data in loader:
# #         data = data.to(device)
# #         optimizer.zero_grad()
# #         out = model(data)
# # #         print(output.shape)
# #         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
# #         loss.backward()
# #         loss_all += loss.item()
# #         optimizer.step()
# #     if epoch % 50 == 0:
# #         print(loss_all)

# def main():
#     train()
# main()


# ##2023/4/23

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


def loss_modularity(r, bin_adj, mod):
    bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()

def draw(z,r):
    colors = [
            '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
            '#ffd700','green']
    
    # 使用TSNE先进行数据降维，形状为[num_nodes, 2]
    z = TSNE(n_components=2).fit_transform(z.detach().numpy())
    getProvince()
    # z = np.c_[z,province_list]
    
    # z.append(province_list)
    # print(z)
    
    r = r.detach().numpy()
    result = np.argmax(r, axis=1)
    
    plt.figure(figsize=(8, 8))
    
    
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


start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = get_logger(start_time)
# logger.get_logger()
# logger.add_handler(start_time)
logger.info("Begin")

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


model_cluster = GCNClusterNet(nfeat=data.x.size(1), nhid=50, nout=50, dropout=0.2, K=K, cluster_temp=50)
optimizer = torch.optim.Adam(model_cluster.parameters(), lr=0.01, weight_decay=5e-4)
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

iter_num = 10001
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
        logger.info(r)
        draw(embeds, r)
 

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
            draw(embeds, r)
            break



# visualize 
# G = to_networkx(data, to_undirected=True)
# visualize_graph(G, color=data.y)

# 可视化节点的embedding

# with torch.no_grad():
#     # 不同类别节点对应的颜色信息
#     colors = [
#             '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
#             '#ffd700'
#         ]

#     model_cluster.eval() # 开启测试模式
#     # 获取节点的embedding向量，形状为[num_nodes, embedding_dim]
#     _, _, z, _ = model_cluster(torch.arange(data.num_nodes, device=device))
#     # 使用TSNE先进行数据降维，形状为[num_nodes, 2]
#     z = TSNE(n_components=2).fit_transform(z.detach().numpy())
#     y = data.y.detach().numpy()

#     plt.figure(figsize=(8, 8))
    
#     # 绘制不同类别的节点
#     for i in range(dataset.num_classes):
#         # z[y==0, 0] 和 z[y==0, 1] 分别代表第一个类的节点的x轴和y轴的坐标
#         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
#     plt.axis('off')
#     plt.show()
