import torch
import math
import os
import time
import copy
import numpy as np
import logging
from metrics import All_Metrics

# index = [0, 2, 6, 9]
# 2t 2rh

def get_logger(root, name=None, debug=True):
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
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
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.cor_index = 0
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        # self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                # data = data[..., :self.args.input_dim]
                # label = target[..., :self.args.output_dim]
                # label = target[...,self.cor_index]
                
                b,h,w,_ = target.shape
                label = target[...,:5].reshape(b,h*w,-1)
                # output = self.model(data, target, teacher_forcing_ratio=0.)

                # b,t,n,c = data.shape
                # data = data.reshape(b,t,58,47,c)
                # data = data.permute(0,1,4,2,3) # convlstm (b, t, c, h, w)

                output = self.model(data)
                if self.args.real_value:
                    # 模型预测的是scale后的值
                    # loss计算的时候使用真实值
                    max_val = torch.Tensor(self.scaler.data_max_).cuda()[...,:5]
                    min_val = torch.Tensor(self.scaler.data_min_).cuda()[...,:5]
                    output = (output * (max_val - min_val)) + min_val
                    # output = self.scaler.inverse_transform(output)
                
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
            # data.shape: [64, 7, 58, 47, 15] scale后的数据
            # features = ['2t', 'tp', '2rh', 'pres', '10vor', '10div','10ws', 'u_10ws', 'v_10ws','100ws', 'u_100ws','v_100ws', 'lon', 'lat', 'dem']
            # target.shape: [64, 58, 47, 12] 没有scale的数据
            # 单个因子
            # label = target[...,self.cor_index]
            # 全部因子
            b,h,w,_ = target.shape
            label = target[...,:5].reshape(b,h*w,-1)
            # b,t,n,c = data.shape
            # data = data.reshape(b,t,58,47,c)
            # data = data.permute(0,1,4,2,3) # convlstm (b, t, c, h, w)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            # if self.args.teacher_forcing:
            #     global_step = (epoch - 1) * self.train_per_epoch + batch_idx
            #     teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            # else:
            #     teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data)
            if self.args.real_value:
                # 模型预测的是scale后的值
                # loss计算的时候使用真实值
                max_val = torch.Tensor(self.scaler.data_max_).cuda()[...,:5]
                min_val = torch.Tensor(self.scaler.data_min_).cuda()[...,:5]
                output = (output * (max_val - min_val)) + min_val
                # output = self.scaler.inverse_transform(output)
            
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
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, self.cor_index)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, cor_index, path=None):
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
                # data = data[..., :args.input_dim]
                # label = target[..., :args.output_dim]
                # label = target[...,cor_index]
            
                b,h,w,_ = target.shape
                label = target[...,:5].reshape(b,h*w,-1)
                # b,t,n,c = data.shape
                # data = data.reshape(b,t,58,47,c)
                # data = data.permute(0,1,4,2,3) # convlstm (b, t, c, h, w)

                output = model(data)
                y_true.append(label)
                y_pred.append(output)
        y_true = torch.cat(y_true, dim=0) # 真实值
        y_pred = torch.cat(y_pred, dim=0) # 预测值
        
        # if args.real_value:
        #     # 模型预测的是scale后的值
        #     max_val = torch.Tensor(scaler.data_max_).cuda()[...,cor_index]
        #     min_val = torch.Tensor(scaler.data_min_).cuda()[...,cor_index]
        #     y_pred = (y_pred * (max_val - min_val)) + min_val
        if args.real_value:
            # 模型预测的是scale后的值
            # loss计算的时候使用真实值
            max_val = torch.Tensor(scaler.data_max_).cuda()[...,:5]
            min_val = torch.Tensor(scaler.data_min_).cuda()[...,:5]
            y_pred = (y_pred * (max_val - min_val)) + min_val
            # output = self.scaler.inverse_transform(output)
        
        
        # np.save('{}/true.npy'.format(args.log_dir), y_true.cpu().numpy())
        # np.save('{}/pred.npy'.format(args.log_dir), y_pred.cpu().numpy())

        for c in range(5):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[...,c], y_true[...,c], args.mae_thresh, args.mape_thresh)
            logger.info("MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape*100))
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