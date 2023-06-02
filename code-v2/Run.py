import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
torch.autograd.set_detect_anomaly(True)

from Hierachy import *
from WeatherGNN import WeatherGNN
from trainer import Trainer
from get_dataloader import *

def init_seed(seed):
    # torch.cuda.cudnn_enabled = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


model_name = 'WeatherGNN'
# parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--model', default='WeatherGNN', type=str, help='model name')
args.add_argument('--data_path', default='/mnt/workspace/NWP_data/data/', type=str, help='data path')
args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')

args.add_argument('--log_step', default=20, type=int)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=5, type=int)

args.add_argument('--debug', default=False, type=eval, help='debug mode')
args.add_argument('--seed', default=42, type=int, help='seed')
args.add_argument('--epochs', default=100, type=int)
args.add_argument('--batch_size', default=64, type=int, help='batch size')
args.add_argument('--grad_norm', default=False, type=eval)
args.add_argument('--max_grad_norm', default=5, type=int)
args.add_argument('--real_value', default=True, type=eval, help='true: 模型预测是scaled后的值')

args.add_argument('--mae_thresh', default=None, type=eval)
args.add_argument('--mape_thresh', default=0.1, type=float)

args.add_argument('--embed_dim', default=10, type=int)
args.add_argument('--hidden_dim', default=32, type=int)

args.add_argument('--k', default=3, type=int)
args.add_argument('--time_len', default=7, type=int)
args.add_argument('--factor_num', default=11, type=int)
args.add_argument('--grid_num', default=31*41, type=int)
args.add_argument('--out_num', default=5, type=int)

args = args.parse_args()
args.level_num = int(math.log(args.grid_num)/math.log(args.k))
log_time = str(int(time.time()))
args.log_dir = f"/mnt/workspace/NWP_model/code-v2/log/{log_time}"

init_seed(args.seed)

supports = torch.zeros((args.grid_num*args.factor_num,args.grid_num*args.factor_num)).to(args.device)
for i in range(args.grid_num):
    supports[i*args.factor_num:(i+1)*args.factor_num,i*args.factor_num:(i+1)*args.factor_num] = torch.ones((args.factor_num,args.factor_num))

# A_init = torch.rand((args.grid_num,args.grid_num))
A_init = get_initial_matrix(args.data_path)
S_set, A_set, W_set, RS_set = get_structure(args.k, args.grid_num, A_init, args.device)

model = WeatherGNN(args).to(args.device)
train_loader, val_loader, test_loader, scaler = get_dataloader(args.batch_size,args.data_path)
loss = torch.nn.L1Loss().to(args.device)
lr_scheduler = None
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.003, eps=1.0e-8, weight_decay=0, amsgrad=False)

trainer = Trainer(model, supports, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, S_set, A_set, W_set, RS_set, lr_scheduler=lr_scheduler)
trainer.train()

print(233)
