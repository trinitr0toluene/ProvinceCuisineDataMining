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
from torch_geometric.utils.convert import to_networkx
from sklearn.manifold import TSNE
from CalModularity import Q
import torch_geometric



# from torch.utils.tensorboard import SummaryWriter


edge_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/edge_features.xlsx'
node_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_features.xlsx'
label_filepath = '/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/node_class_new.xlsx'
k = 3

K = 3

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
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=1)
    
    global label
    
    label = torch.tensor(df['class'], dtype=torch.int64)
    print(label)
     
    
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

dataset = MyOwnDataset(root='/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/2stage-sheet2')

# print(dataset.num_classes) # 0
# print(dataset[0].num_nodes) # 31
# print(dataset[0].num_edges) # 93
# print(dataset[0].num_features) # 8

data_list = [dataset[0]]
data = dataset[0]
# print(data)
# print(type(data))

G = to_networkx(data)

# def get_dis(data, center):
#     ret = []
#     print(len(data))
#     for point in data:
#         # np.tile(a, (2, 1))就是把a先沿x轴复制1倍，即没有复制，仍然是[0, 1, 2]。 再把结果沿y方向复制2倍得到array([[0, 1, 2], [0, 1, 2]])
#         # k个中心点，所以有k行
#         diff = np.tile(point, (K, 1)) - center
#         squaredDiff = diff ** 2  # 平方
#         squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
#         print(f"sq: {squaredDist}")
        
#         distance = squaredDist ** 0.5  # 开根号
#         ret.append(distance)
#     print(f"get_dis: {np.array(ret)}")
#     print(len(ret))    
#     return np.array(ret)

# def edge_index_to_coo_adj(edge_index):
#     x_sh=edge_index.shape[0]
#     edge_shape= np.zeros((x_sh,x_sh)).shape
#     value = torch.FloatTensor(np.ones(edge_index.shape[1]))
#     adj = torch.sparse_coo_tensor(edge_index,value,edge_shape)
#     adj = adj.toarray()
#     return adj

def calEuclidean(data, center):
    dist = np.sqrt(np.sum(np.square(data-center))) 
    # print(type(dist))
    return dist

def k_means(m, K):
    """
    :param m: GCN的训练结果
    :param K: 类别数
    :return: 聚类结果
    """
    # print(m)
    nodes = list(G.nodes)
    # 任意选择K个节点作为初始聚类中心
    centers = []
    temp = []
    for i in range(K):
        t = np.random.randint(0, len(nodes) - 1)
        # print(nodes[t])
        # print(m[nodes[t]])
        # print(centers)
        # if m[nodes[t]] not in centers:
        if nodes[t] not in temp:
            temp.append(nodes[t])
            centers.append(m[nodes[t]])  # 中心为8维向量

    # 迭代50次
    res = {}
    for i in range(K):
        res[i] = []

    for time in range(50):
        # clear
        for i in range(K):
            res[i].clear()
        # 算出每个点的向量到聚类中心的距离
        nodes_distance = {}
        for node in nodes:
            # node到中心节点的距离
            node_distance = []
            for center in centers:
                # print('m[node]:',node, type(m[node]))
                # print('center:', center, type(center))
                node_distance.append(calEuclidean(m[node], center))  # get_dis函数计算单个节点到单个center的欧式距离
            nodes_distance[node] = node_distance  # 保存node节点到各个中心的距离
        # 对每个节点重新划分类别，选择一个最近的节点进行分类，类别为0-5
        # print(i)
        for node in nodes:
            temp = nodes_distance[node]  # 存放着3个距离

            # print(temp)
            # print(min(temp))
            cls = temp.index(min(temp))
            res[cls].append(node)

        # 更新聚类中心
        centers.clear()
        for i in range(K):
            center = []
            for j in range(dataset.num_classes):
                t = [m[node][j] for node in res[i]]  # 第i个类别中所有node节点的第j个坐标
                center.append(np.mean(t))
            centers.append(center)
           

    return res
# transform = RandomLinkSplit(is_undirected=True)
# print(type(data))
# data = transform(data)
# data = train_test_split_edges(data)
# print(data)
# print(type(data))
# print(data.x)
# loader = DataLoader(data_list, batch_size=1)
def draw(z,r):
    colors = [
            '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
            '#ffd700','green']
    
    # 使用TSNE先进行数据降维，形状为[num_nodes, 2]
    z = TSNE(n_components=2).fit_transform(z)
    province_list = getProvince()
    # z = np.c_[z,province_list]
    
    # z.append(province_list)
    # print(z)
    
    result = r
    # print(type(result))
    plt.figure(figsize=(8, 8))
    
    # print(result.get(1))
    # print(z[result.get(1),0])
    for j in range(K):
        plt.scatter(z[result.get(j),0], z[result.get(j),1],s=450, color=colors[j], alpha=0.5)
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
    plt.savefig('/home/zhangziyi/code/ProvinceCuisineDataMining/Log/'+start_time[:10]+'/'+start_time[11:]+'/GCN-2stage')    

start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = get_logger(start_time)
# logger.get_logger()
# logger.add_handler(start_time)
logger.info("Begin")
logger.info(f'label:{label}')

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
        x = F.log_softmax(x, dim=-1) 
        # print(x)
        return x   #若不对输出数据做归一化处理，则loss为负值(二维张量：dim=1和dim=-1结果相同)
        #x为节点的embedding

 
    
# 5.3) 训练 & 测试.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)
model = Net().to(device)

lr = 0.01
wd = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
logger.info(f'lr = {lr}  weight_decay:{wd}')
# @torch.no_grad()   #不需要计算梯度，也不进行反向传播

iter_num = 101  
for epoch in range(iter_num):
    optimizer.zero_grad()#清空所有被优化的变量的梯度
    model.train()#设置成train模式
    out = model()
    # print('out:',out)
    label = data.y
    # one_hot = F.one_hot(label, num_classes = dataset.num_classes)
    # print(one_hot)
    
    loss = F.nll_loss(model()[data.train_mask], label[data.train_mask])
    loss.requires_grad_(True)
    loss.backward()
     #损失函数训练参数用于节点分类
    optimizer.step()#步长
     
    logger.info(f'epoch{epoch + 1}   loss:{loss}')
    print(f'epoch{epoch + 1}   loss:{loss}')
    if epoch == iter_num-1:
        out = out.detach().cpu().numpy()
        result = k_means(out, K)
        logger.info(f'Final Result:{result}')
        print(f'Final Result:{result}')
        # print(data.edge_index.cpu())
        # adj_array = edge_index_to_adj(data.edge_index.cpu())

        adj_array = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index)
        adj_array = adj_array.toarray()
        # print(adj_array)
        #计算模块度
        score = Q(adj_array, result)
        print(f'模块度为：{score}')
        logger.info(f'模块度为：{score}')
        draw(out,result)
        # print(data.y)
    


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

# graph = nx.Graph()
# edge_index = data.edge_index.T

# for i in range(edge_index.shape[0]):
#     graph.add_edge(edge_index[i][0].item(), edge_index[i][1].item())

# pos = nx.kamada_kawai_layout(graph)



# if __name__=='__main__':
#     K_means()

