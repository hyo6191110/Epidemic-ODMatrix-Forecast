import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from time import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys

from utils.dataset import HumanMobilityDataset
from utils.data_process import diag_zero, y_dim_reduce

from models.GCN_3D import GCN_3D
from models.GCN_Transformer import GCN_Transformer
from models.PureTransformer import PureTransformer
from models.GCN_OD_Transformer import GCN_OD_Transformer
from models.GCN_n_Transformer import GCN_n_Transformer
from models.GCformer_nopatch import GCformer_nopatch
from utils.data_process import *


def main():
    parser = argparse.ArgumentParser(description='Human Mobility Pred')
    parser.add_argument('--seed', type=int, default=12345, metavar='S',
                        help='random seed')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DO')
    parser.add_argument('--T_in', type=int, default=2, metavar='TI',
                        help='number of input time intervals in a training sample')
    parser.add_argument('--T_out', type=int, default=2, metavar='TO',
                        help='number of output time intervals in a training sample')
    parser.add_argument('--T_label', type=int, default=1, metavar='TL',
                        help='number of labeled time intervals in out time intervals')
    parser.add_argument('--load', type=int, default='0', metavar='LD',
                        help='load checkpoint/epoch_x.pt (x>0)')
    parser.add_argument('--normalize', type=int, default=0,
                        help='way of normalize (0-no,1--MinMax,2--Log,3--Mul,4--Add)')
    parser.add_argument('--norm_param', type=float, default=1,
                        help='normalize 3,4 param')
    parser.add_argument('--data', type=str, default='JHT',
                        help='location of the data file')
    parser.add_argument('--data_X', type=str, default='OD',
                        help='value: ioFlow(C=2)/OD(C=V)')
    parser.add_argument('--patch_size', type=int, default=512, help='patch size')
    parser.add_argument('--patch_method', type=str, default='embed', metavar='PM', help='embed / divide')
    parser.add_argument('--fusion_method', type=str, default='concat', metavar='FM',
                        help='concat / add / mul / biLinear / sum-pooling')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--cuda', type=int, default=0, help='cuda')
    parser.add_argument('--model_store', action='store_true', help='save model', default=False)
    parser.add_argument('--writer_store', action='store_true', help='save model', default=False)
    parser.add_argument('--zero_diag', action='store_true', help='save model', default=False)
    args = parser.parse_args()
    # seed设置
    torch.manual_seed(args.seed)
    # cuda设置
    if torch.cuda.is_available():
        device = torch.device('cuda', args.cuda)
    else:
        device = torch.device("cpu")

    # 参数设置
    V = 0
    T_in = 0
    T_out = 0
    if args.data=='JHT':
        V = 47
        T_in = 7
        T_out = 28
    elif args.data=='WX':
        V = 92
        T_in = 12
        T_out = 24
    elif args.data=='EZ':
        V = 26
        T_in = 12
        T_out = 24
    T_label=T_in
    scaler_X = LogScaler()
    # 路径
    LOAD_MODEL='D:\Projects\PycharmProjects\HumanFlow\HMPred-Graph\data\{}\\best_{}.pt'.format(args.data,args.data)
    OD_PATH='D:\Projects\PycharmProjects\HumanFlow\HMPred-Graph\data\{}\OD_{}.npy'.format(args.data,args.data)
    A_PATH='D:\Projects\PycharmProjects\HumanFlow\HMPred-Graph\data\{}\OD_{}.npy'.format(args.data,args.data)
    TIMEF_PATH='D:\Projects\PycharmProjects\HumanFlow\HMPred-Graph\data\{}\\timeF_{}.npy'.format(args.data,args.data)

    # 建立模型
    model = GCN_Transformer(C=V, V=V, device=device,
                                   patch_size=args.patch_size,
                                   d_model=args.d_model,
                                   d_ff=args.d_ff,
                                   T_i=T_in,
                                   T_o=T_out,
                                   T_label_i=T_label,
                                   data=args.data,
                                   fusion=args.fusion_method,
                                   patch=args.patch_method,
                                   dropout=args.dropout)
    model = model.to(device)
    # 加载模型
    model.load_state_dict(torch.load(LOAD_MODEL))

    # 预测数据准备
    train_OD = np.load(OD_PATH)
    x = train_OD[-T_in-T_out:-T_out]
    y = train_OD[-T_out:]
    if args.data != 'JHT':
        x = scaler_X.fit_transform(x)
    else:
        y = scaler_X.inverse_transform(y).astype(np.int64)
    A_raw = np.load(A_PATH)
    A_raw = A_raw[-T_in-T_out:-T_out]
    A = np.zeros((T_in, 2, V, V), dtype=np.float32)
    add_loop = False
    dataset_need_add_loop = ['NYC', 'NYC15m', 'NYC60m', 'WX', 'EZ']
    if args.data in dataset_need_add_loop:
        add_loop = True
    A[:, 0, :, :] = sym_norm_laplacian(A_raw, deg_axis=0, add_loop=add_loop)
    A[:, 1, :, :] = sym_norm_laplacian(A_raw, deg_axis=1, add_loop=add_loop)
    timeF = np.load(TIMEF_PATH)
    x_timeF = timeF[-T_in-T_out:-T_out]
    y_timeF = timeF[-T_out:]
    # 模型格式适配
    x = torch.from_numpy(x.astype(np.float32)).to(device).unsqueeze(0)
    A = torch.from_numpy(A.astype(np.float32)).to(device).unsqueeze(0)
    x_timeF = torch.from_numpy(x_timeF.astype(np.int32)).to(device).unsqueeze(0)
    y_timeF = torch.from_numpy(y_timeF.astype(np.int32)).to(device).unsqueeze(0)

    # 进行一次预测
    model.eval()
    y_pred = model(x, A, x_timeF, y_timeF)

    # 数据转换
    y_pred=y_pred.squeeze(0)
    y_pred=y_pred.cpu().detach().numpy()
    y_pred = scaler_X.inverse_transform(y_pred).astype(np.int64)

    # 检查结果
    for i in range(len(y_pred[0])):
        print(y_pred[0][i])
    for i in range(len(y[0])):
        print(y[0][i])
        
    # 保存结果
    print(y_pred.shape)
    np.save('OD_pred_JHT.npy', y_pred)


if __name__ == '__main__':
    main()
