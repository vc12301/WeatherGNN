import torch
import math
import os
import time
import copy
import numpy as np
import logging
import torch.nn as nn
from metrics import All_Metrics


def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')


def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger


class Trainer(object):
    def __init__(self, model, supports, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, S_set, A_set, W_set, RS_set, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.cor_index = 0
        self.model = model
        self.supports = supports
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.scaler = scaler
        # self.max_val = args.data_max
        # self.min_val = args.data_min

        self.args = args
        self.S_set = S_set
        self.A_set = A_set
        self.W_set = W_set
        self.RS_set = RS_set        
        self.lr_scheduler = lr_scheduler

        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)
            
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        print_model_parameters(model, only_num=False)            

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                label = target[...,:5]
                output = self.model(data, self.S_set, self.RS_set, self.A_set, self.W_set, self.supports)
                if self.args.real_value:
                    # 模型预测的是scale后的值
                    # loss计算的时候使用真实值
                    # max_val = torch.Tensor(self.scaler.data_max_).cuda()[...,:5]
                    # min_val = torch.Tensor(self.scaler.data_min_).cuda()[...,:5]
                    # output = (output * (max_val - min_val)) + min_val
                    output = self.scaler.inverse_transform(output)
                
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # weather features: ['100 metre U wind component', '100 metre V wind component','10 metre U wind component','10 metre V wind component','2 metre temperature','Mean sea level pressure','Surface pressure','Total precipitation']
            # geo featrues: [lan, lon, dem]
            # data.shape: [64, 7, 1271, 11] scale后的数据 B,T,N,F
            # target.shape: [64, 1271, 5] 没有scale的数据
            label = target[...,:5]
            self.optimizer.zero_grad()

            output = self.model(data, self.S_set, self.RS_set, self.A_set, self.W_set, self.supports)
            if self.args.real_value:
                # 模型预测的是scale后的值
                # loss计算的时候使用真实值

                output = self.scaler.inverse_transform(output)
            
            loss = self.loss(output, label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

        #learning rate decay
        # if self.args.lr_decay:
        #     self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        print('train start!')
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, self.cor_index,self.S_set, self.RS_set, self.A_set, self.W_set, self.supports)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, cor_index, S_set, RS_set, A_set, W_set, supports, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                label = target[...,:5]
                output = model(data, S_set, RS_set, A_set, W_set, supports)
                y_true.append(label)
                y_pred.append(output)
        y_true = torch.cat(y_true, dim=0) # 真实值
        y_pred = torch.cat(y_pred, dim=0) # 预测值

        if args.real_value:
            # 模型预测的是scale后的值
            # loss计算的时候使用真实值
            output = scaler.inverse_transform(output)
        
        
        # np.save('{}/true.npy'.format(args.log_dir), y_true.cpu().numpy())
        # np.save('{}/pred.npy'.format(args.log_dir), y_pred.cpu().numpy())

        for c in range(5):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[...,c], y_true[...,c], args.mae_thresh, args.mape_thresh)
            logger.info("MAE: {:.2f}, RMSE: {:.2f}".format(mae, rmse))
        # mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        # logger.info("MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape*100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))