import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities as lpa
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset
from log import get_logger
import datetime
from torch_geometric.utils.convert import to_networkx
import torch_geometric
from CalModularity import Q



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
    
    label = torch.tensor(df_new['max_idx'], dtype=torch.int64)
     
    
    # print(label)
    # print(label.size())
    return label


def getProvince():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=0)
    global province_list
    province_list = df.values[: , 0]
    return province_list



def draw(z,r):
    colors = [
            'pink','orange','r','g','b','y','m','gray','c','brown', '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
            '#ffd700']
    
    province_list = getProvince()

    K = len(r)
    # z = z.values()
    z = np.array(list(z.values()))
    # print(z)
    result = dict()
    for index, item in enumerate(com):
        result[index] = list(item)
    # print(result)
    # result = dict(enumerate(com))
    # print(f'type of result:{type(result)}'  f'result:{result}')
    # print(f'type of z:{type(z)}'  f'result:{z}')
    plt.figure(figsize=(8, 8))
    
    # print(result.get(1))
    # print(z[result.get(1),0])
    for j in range(K):
        plt.scatter(z[result.get(j),0], z[result.get(j),1],s=450, color=colors[j], alpha=0.5)
    for i in range(z.shape[0]):  ## for every node
        plt.annotate(province_list[i], xy = (z[i,0],z[i,1]),  xytext=(-20, 10), textcoords = 'offset points',ha = 'center', va = 'top')    

    plt.axis('off')
    plt.show()
    plt.savefig('/home/zhangziyi/code/ProvinceCuisineDataMining/Log/'+start_time[:10]+'/'+start_time[11:]+'/LPA_scatter')    

def calculate(result):
        adj_array = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index)
        adj_array = adj_array.toarray()
        # print(adj_array)
        #计算模块度
        score = Q(adj_array, result)
        print(f'模块度为：{score}')
        logger.info(f'模块度为：{score}')

# build the PyG Dataset


 
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
                    )

        data_list = [data]
        
 
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
 
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = MyOwnDataset(root='/home/zhangziyi/code/ProvinceCuisineDataMining/dataset/LPA')

data_list = [dataset[0]]
data = dataset[0]


start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = get_logger(start_time)
logger.info("Begin")



G = to_networkx(data)


com = list(lpa(G))
print(f'社区数量:{len(com)}'  f'聚类结果：{com}')
logger.info(f'社区数量:{len(com)}'  f'聚类结果：{com}')

# 下面是画图
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
for i in range(len(com)):
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=com[i], 
                           node_color = color_list[i+2],  
                           label=True)
plt.show()
plt.savefig('/home/zhangziyi/code/ProvinceCuisineDataMining/Log/'+start_time[:10]+'/'+start_time[11:]+'/LPA_nx')

draw(pos, com)

calculate(com)

