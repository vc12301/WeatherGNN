import numpy as np

from tqdm import tqdm
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def lat_long_dis_cal(lat1,lat2,lon1,lon2):
    return geodesic((lat1,lon1),(lat2,lon2)).m

def dem_dis_cal(dem1,dem2):
    return np.abs(dem1-dem2)


def geo_dis(sp_dis,dem_dis,lamb=1,mu=1):
    return np.sqrt(lamb*sp_dis**2+mu*dem_dis**2)

def gaussian_dis(x, sigma=0.1, epsilon=0.5):
    n = x.shape[0]
    w = np.ones([n, n]) - np.identity(n)
    return (np.exp((-x ** 2) / (2 * sigma**2)) >= epsilon)*w


def calculated_distance_matrix(num):
    data_path = '/home/wbq/nwp/data/'
    geo_info = np.load(f'{data_path}geo_info.npy')
    
    shuiping_dis = np.zeros((num,num))
    for i in tqdm(range(num)):
        for j in range(i,num):
            # print(geo_info[i],geo_info[j])
            shuiping_dis[i,j] = lat_long_dis_cal(
                geo_info[i][0],geo_info[j][0],
                geo_info[i][1],geo_info[j][1]
            )
    shuiping_dis_scaler = MinMaxScaler().fit(shuiping_dis)
    shuiping_dis_ = shuiping_dis_scaler.transform(shuiping_dis)

    chuizhi_dis = np.zeros((num,num))
    for i in tqdm(range(num)):
        for j in range(i,num):
            # print(geo_info[i],geo_info[j])
            chuizhi_dis[i,j] = dem_dis_cal(
                geo_info[i][2],geo_info[j][2]
            )
    chuizhi_dis_scaler = MinMaxScaler().fit(chuizhi_dis)
    chuizhi_dis_ = chuizhi_dis_scaler.transform(chuizhi_dis)

    # shuiping_dis = np.load('{data_path}shuiping_dis.npy')
    # chuizhi_dis = np.load('{data_path}chuizhi_dis.npy')
    
    np.save('{data_path}shuiping_dis.npy',shuiping_dis_)
    np.save('{data_path}chuizhi_dis.npy',chuizhi_dis_)

    geo_d = np.zeros((num,num))
    for i in tqdm(range(num)):
        for j in range(i,num):
            geo_d[i,j] = geo_dis(shuiping_dis_[i,j],chuizhi_dis_[i,j])
    geo_d_ = geo_d+geo_d.T
    
    np.save('{data_path}distance.npy',geo_d_)
    
    return geo_d_
    

def get_distance_matrix(path):
    distance_matrix = np.load(path)
    return distance_matrix

def get_structure(A_1, layer_1_node_num, layer_2_node_num, layer_3_node_num):
    # 第一层到第二层的分配矩阵
    assignment_labels_1 = KMeans(n_clusters=layer_2_node_num, random_state=0, n_init=10).fit(A_1).labels_.reshape(-1,1)
    S_1 = OneHotEncoder(handle_unknown='ignore').fit_transform(assignment_labels_1).toarray()
    # 第二层静态邻接矩阵
    A_2 = S_1.T@A_1@S_1
    # 第二层到第三层的分配矩阵
    assignment_labels_2 = KMeans(n_clusters=layer_3_node_num, random_state=0, n_init=10).fit(A_2).labels_.reshape(-1,1)
    S_2 = OneHotEncoder(handle_unknown='ignore').fit_transform(assignment_labels_2).toarray()
    # 第三层静态邻接矩阵
    A_3 = S_2.T@A_2@S_2

    S_set = [S_1,S_2]
    A_sets = [A_1,A_2,A_3]

    # 第二层的祖先节点的邻居
    neighbor_index_layer_2 = []
    for i in range(layer_2_node_num):
        # 第二层节点属于第三层的哪一祖先节点
        ancestor_index_i = np.nonzero(S_2[i,:])[0].tolist()
        # 第三层对应的祖先节点包含第二层哪些节点
        neighbor_index_i = np.nonzero(S_2[:,ancestor_index_i])[0].tolist()
        # 把自身去掉
        neighbor_index_i.remove(i)
        neighbor_index_layer_2.append(neighbor_index_i)

    # 第三层的祖先邻居
    neighbor_index_layer_3 = []
    for i in range(layer_3_node_num):
        tmp = list(range(layer_3_node_num))
        tmp.pop(i)
        neighbor_index_layer_3.append(tmp)
    
    return S_set, A_sets, neighbor_index_layer_2, neighbor_index_layer_3


