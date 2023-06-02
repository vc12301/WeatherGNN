import numpy as np
import pandas as pd

from multiprocessing.pool import ThreadPool
from tslearn.metrics import dtw
from tqdm import tqdm

def task(i,j):
    print(i,j)
    grids_dtw[i,j] = dtw(grids[:,i],grids[:,j])


save_path = '/home/wbq/WeatherGNN_code/data'
grids_num = 1271

data = np.load('/home/wbq/WeatherGNN_code/data/ningxia_nwp_train.npy')
data = data.reshape(8760,-1,11)

training_len = int(8760*0.7)
# need_col = ['100 metre U wind component', '100 metre V wind component','10 metre U wind component','10 metre V wind component','2 metre temperature','Mean sea level pressure','Surface pressure','Total precipitation']
grids = data[:training_len,:,2]
grids_dtw = np.zeros((grids_num,grids_num))

with ThreadPool(100) as pool:
    async_results = [pool.apply_async(task, args=(i, j)) for i in range(grids_num) for j in range(i+1,grids_num)]
    results = [ar.get() for ar in async_results]
    np.save(f'{save_path}/dtw_matrix_10ws_U.npy',grids_dtw)