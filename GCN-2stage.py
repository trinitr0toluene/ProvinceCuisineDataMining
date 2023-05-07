from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import heapq

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from log import get_logger
import datetime

# from torch.utils.tensorboard import SummaryWriter


edge_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/edge_features.xlsx'
node_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_features.xlsx'
label_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_class.xlsx'
k = 3

#initialize
edge_index = []
start = []
end = []

edge_attr = []
node_feature = []
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

    return edge_index, edge_attr


def excel2node():
    df = pd.read_excel(f'{node_filepath}', engine='openpyxl', sheet_name=0)  

    print(df.shape)
    rows = df.shape[0]
    columns = df.shape[1]

    print(rows)
    print(columns)

    df_new = df.drop(df.columns[[0,1]], axis=1)
    print(df_new.shape)
    global node_feature
    node_feature = torch.tensor(df_new.values, dtype=torch.float)

    # print(node_feature)
    print(node_feature.size())

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
    
    label = torch.tensor(df_new['max_idx'], dtype=torch.int64)
     
    
    # print(label)
    # print(label.size())
    return label


def getProvince():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=0)
    global province_list
    province_list = df.values[: , 0]
    return province_list

def getMask():
    mask = []
    for i in range(31):
        mask.append(True)
    mask = torch.tensor(mask)
        
    
    # print(df_new['max_idx'])
    
    
    
     
    
    # print(label)
    # print(label.size())
    return mask
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
                    y=addClass(),
                    train_mask=getMask())

        data_list = [data]
        
 
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
 
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = MyOwnDataset(root='/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/2stage')

print(dataset.num_classes) # 0
print(dataset[0].num_nodes) # 31
print(dataset[0].num_edges) # 93
print(dataset[0].num_features) # 8

data_list = [dataset[0]]
data = dataset[0]
print(data)
print(type(data))

# transform = RandomLinkSplit(is_undirected=True)
# print(type(data))
# data = transform(data)
# data = train_test_split_edges(data)
# print(data)
# print(type(data))
# print(data.x)
# loader = DataLoader(data_list, batch_size=1)
hidden_dim = 16

#  定义2层GCN的网络.
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset.num_classes)
    
    
    def forward(self):
        x, edge_index, edge_weight = data.x, torch.tensor(data.edge_index, dtype=torch.int), torch.tensor(data.edge_attr, dtype=torch.float)  #赋值data.x特征向量edge_index图的形状，edge_attr权重矩阵


        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)   
        #x,edge_index,edge_weight特征矩阵，邻接矩阵，权重矩阵组成GCN核心公式
        x = F.dropout(x, training=self.training)   #用dropout函数防止过拟合
        x = self.conv2(x, edge_index, edge_weight)  #输出
        # print(x)
        return x
        #x为节点的embedding

 
    
# 5.3) 训练 & 测试.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# @torch.no_grad()   #不需要计算梯度，也不进行反向传播

model.train()
    
for epoch in range(200):
    optimizer.zero_grad()#清空所有被优化的变量的梯度
    model.train()#设置成train模式
    label = data.y
    # one_hot = F.one_hot(label, num_classes = dataset.num_classes)
    # print(one_hot)
    loss = F.nll_loss(model()[data.train_mask], label[data.train_mask])
    loss.requires_grad_(True)
    loss.backward()
     #损失函数训练参数用于节点分类
    optimizer.step()#步长
     
    
    print(f'epoch{epoch + 1}   loss:{loss}')
    
    



# for epoch in range(200):
#     loss_all = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data)
# #         print(output.shape)
#         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].long())
#         loss.backward()
#         loss_all += loss.item()
#         optimizer.step()
#     if epoch % 50 == 0:
#         print(loss_all)

# def main():
#     train()
# main()


# Visulization module
# # 将整个数据集可视化出来 
# g = nx.DiGraph() # 建一个空的有向图
# name,edgeinfo = edge
# src = edgeinfo[0].numpy()
# dst = edgeinfo[1].numpy()
# edgelist = zip(src,dst)
# for i,j in edgelist:
#     g.add_edge(i,j) 
# plt.rcParams['figure.dpi'] = 300 #分辨率
# fig , ax1 = plt.subplots(figsize=(10,10))
# nx.draw_networkx(g , ax = ax1 , font_size=6 , node_size = 150)
# plt.show()


