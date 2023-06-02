import os
import numpy as np
import pickle
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.utils.data

def re_shape_x(x_data):
    n,t,h,w,x_c = x_data.shape
    x_data = x_data.transpose(0,2,3,1,4)
    x_data = x_data.reshape(n*h*w,t,x_c)
    return x_data


def re_shape_y(y_data):
    n,h,w,y_c = y_data.shape
    y_data = y_data.reshape(n*h*w,y_c)
    return y_data


def get_dataset():
    data_path = "/mnt/workspace/NWP_data/data/"
    feature_num = 8
    back_window = 3
    ahead_window = 3
    
    raw_x = np.load(data_path+'ningxia_nwp_train.npy').reshape(-1,11)
    raw_y = np.load(data_path+'ningxia_real_train.npy').reshape(-1,8)
    
    # norm
    # scaled_data_x = []
    scaler_x = MinMaxScaler().fit(raw_x)
    scaled_data_x = scaler_x.transform(raw_x)
    # pickle.dump(scaler_x, open(data_path+'scaler_minmax.pkl','wb'))

    reshape_scaled_data_x = scaled_data_x.reshape(-1,31,41,feature_num+3) # T,H,W,D
    reshape_data_y = raw_y.reshape(-1,31,41,feature_num) # T,H,W,D

    train_offset = np.sort(np.concatenate((np.arange(-back_window, ahead_window+1, 1),)))
    # array([-3, -2, -1,  0,  1,  2,  3]) 前后3天

    # get samples
    x, y = [], []
    for t in range(back_window, len(reshape_data_y)-ahead_window):
        x_t = reshape_scaled_data_x[t + train_offset, ...]
        y_t = reshape_data_y[t, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print("x shape: ", x.shape, ", y shape: ", y.shape)


    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    # 如果是MLP、GRU等 不考虑空间关系的 h和w需要和样本数量打通
    # 或者是AGCRN中只考虑 因子图的
    # x_train = re_shape_x(x_train)
    # x_val = re_shape_x(x_val)
    # x_test = re_shape_x(x_test)
    
    # y_train = re_shape_y(y_train)
    # y_val = re_shape_y(y_val)
    # y_test = re_shape_y(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler_x



def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(batch_size):
    # scaler = pickle.load(open(data_path+'scaler_minmax.pkl', 'rb'))
    
    # data = {}
    # for category in ['train', 'val', 'test']:
    #     cat_data = np.load(os.path.join(data_path, category + '.npz'))
    #     data['x_' + category] = cat_data['x']
    #     # 只算风速
    #     data['y_' + category] = cat_data['y'][...,-2]
        # data['y_' + category] = cat_data['y']

    # x_train, y_train, x_val, y_val, x_test, y_test = data['x_train'],data['y_train'],data['x_val'],data['y_val'],data['x_test'],data['y_test']
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_dataset()
    print('Train: ', x_train.shape, y_train.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    
    # get dataloader
    train_dataloader = data_loader(x_train, y_train, batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler
    # 对于scaler的使用 scaler.inverse_transform(train_x['x'][:,0,:,-2])

# start_time = time.time()
# get_dataloader(64)
# end_time = time.time()
# print(end_time-start_time)
