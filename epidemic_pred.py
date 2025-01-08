import numpy as np
import pandas as pd
import time
import pyswarms as ps
import math
import argparse
from sklearn.metrics import r2_score


# 数据准备
#################################################################
# 参数设置
parser = argparse.ArgumentParser(description='Epidemic Pred')
parser.add_argument('--data', type=str, default='WX', help='location of the data file')
parser.add_argument('--show_train', action='store_true', help='show train result', default=False)
parser.add_argument('--show_seir', action='store_true', help='show seir', default=False)
parser.add_argument('--test_opt', action='store_true', help='optimize while test', default=False)
args = parser.parse_args()


# 数据集
# JHT: R=47 T=406 (2020/1/20-2021/2/28)
# interval = 7 batch = 54:4
# WX: R=92 T=30*24 (2023/6/1-2023/6/30)
# interval = 24 batch = 23:7
# EZ: R=27 T=30*24 (2023/6/1-2023/6/30)
# interval = 24 batch = 23:7
DATASET = args.data
# 结果保存名
NAME = DATASET

# 训练参数保存和数据的路径
BASE_SAVE = 'checkpoint/checkpoint_epidemic/' + NAME
BASE_DATA = 'data/data_epidemic'

# TRAIN/TEST SPLIT
# 训练集进行PSO优化的时间间隔
T_INTERVAL = 7
# 留给训练集多少个T_INTERVAL，后面的都是测试集
TRAIN_BATCH = 54
if DATASET == 'JHT':
    T_INTERVAL = 7
    TRAIN_BATCH = 54
elif DATASET == 'WX':
    T_INTERVAL = 24
    TRAIN_BATCH = 23
elif DATASET == 'EZ':
    T_INTERVAL = 24
    TRAIN_BATCH = 23


# 路径
POPULATION_PATH = BASE_DATA + '/{}/population_{}.npy'.format(DATASET, DATASET)
INFECT_PATH = BASE_DATA + '/{}/infect_{}.npy'.format(DATASET, DATASET)

# 区域人口数据 [R]
population = np.load(POPULATION_PATH)
# 区域感染数据 [T,R]
infection = np.load(INFECT_PATH)
train_infection = infection[:T_INTERVAL * TRAIN_BATCH]
test_infection = infection[T_INTERVAL * TRAIN_BATCH:]
# 区域OD数据 [T,R,R]


# 区域数
num_regions = len(infection[0])
# 区域列表 每个代表区域的gid
regions = range(num_regions)

# SEIR常量超参数
# MIU = mortality rate
MIU = 0.0
# EPSILON = progression rate to infectious state
EPSILON = 0.2
# GAMMA = recovery rate
GAMMA = 0.1
# SIGMA = actual inter-region transition rate
SIGMA = 0.1

###################################################################


# 进行one step的SEIR的更新
# df:目标区域的step t 的SEIR矩阵 [R,4]
# df_flow:目标区域的OD矩阵 [1,R,R]
# beta:传播率的参数，随区域变化[R]
# 输出:目标区域的step t+1 的SEIR矩阵
def SEIR_Single(df, od_flow, beta):
    # df_flow:每个区域人口向其他区域的移动概率分布
    od_flow = od_flow.apply(lambda x: x / max(sum(x), 1), axis=1)
    for i in range(len(od_flow)):
        od_flow.iloc[i, i] = 0
    # the outflow of SEIR of each region
    df_out = od_flow.apply(lambda x: sum(x), axis=1)
    df_dec = df.apply(lambda x: (x*df_out), axis=0)
    # update 1
    df['N'] = df['S'] + df['E'] + df['I'] + df['R']
    df['Beta'] = beta
    df['S'] = df['S'] + MIU * (df['N'] - df['S']) - df['Beta']*df['I']*df['S']/df['N']
    df['E'] = df['E'] + df['Beta'] * df['I'] * df['S'] / df['N'] - (MIU + EPSILON) * df['E']
    df['I'] = df['I'] + EPSILON * df['E'] - (GAMMA + MIU) * df['I']
    df['R'] = df['R'] + GAMMA * df['I'] - MIU * df['R']
    df = df.drop(columns=['N', 'Beta'])
    # the inflow of SEIR coming from other regions
    df_inc = pd.DataFrame(np.matmul(od_flow.T.values, df.values), index=regions, columns=['S', 'E', 'I', 'R'])
    # update 2
    df = df + (df_inc - df_dec) * SIGMA
    return df


# 进行multi step的SEIR的更新
def SEIR_Multi(beta, pred_steps, df_init, od_flows):
    # 初始化SEIR矩阵
    df = df_init.copy()
    prediction = []
    for i in pred_steps:
        # 获取区域的OD矩阵
        od_flow = pd.DataFrame(od_flows[i, :, :], index=regions, columns=regions)
        # update one step
        df = SEIR_Single(df, od_flow, beta)
        # 把SEIR转换成整数
        df = df.fillna(0)
        df_seir_r = df.round().astype(int)
        # 形成感染人数的预测结果
        prediction.append(df_seir_r['I'].tolist())
    return df, np.array(prediction)


# 训练中使用的损失函数
def loss(pred, truth):
    #truth = truth.clip(min=1)
    mape_lambda = 300
    return np.mean(np.abs(pred - truth)) + np.mean(np.abs(pred - truth) / truth.clip(min=1)) * mape_lambda


# 训练中使用的优化函数
def opt_func(X, pred_steps, df_seir, od_flows, infects):
    n_particles = X.shape[0]  # number of particles
    ground_truth = infects[pred_steps, :]
    dist = []
    for i in range(n_particles):
        _, prediction = SEIR_Multi(X[i, :], pred_steps, df_seir, od_flows)
        dist.append(loss(prediction, ground_truth))
    return np.array(dist)


# 在一个长度为T_INTERVAL的训练批次进行PSO优化
def run_interval_PSO(pred_steps, df_seir, od_flows, infects):
    # print('Start Particle Swarm Optimization ...', time.ctime())
    # 超参数设置
    cpu_num = 10
    swarm_size = 50
    iters_num = 100
    beta_low = 0.01
    beta_high = 0.4
    # Dimension of X
    dim = num_regions
    options = {'c1': 0.5, 'c2':0.5, 'w':0.5}
    constraints = (np.array([beta_low] * dim), np.array([beta_high] * dim))
    # 优化器设置
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dim, options=options, bounds=constraints)
    # 进行一次PSO优化
    cost, pos = optimizer.optimize(opt_func, iters=iters_num, n_processes=cpu_num, verbose=False,
                                   pred_steps=pred_steps, df_seir=df_seir, od_flows=od_flows, infects=infects)
    return cost, pos


# beta参数优化
# 以T_INTERVAL划分批次，优化beta以及得出在训练集开始时的初始seir矩阵
def run_all_PSO(od_flows, infects):
    assert len(od_flows) == len(infects), 'len of od_flows and infects should be equal'
    T_train = len(od_flows)
    pred_steps_all = np.arange(T_train)
    # 定义初始的seir矩阵
    df_seir = pd.DataFrame(np.zeros((num_regions, len(['S', 'E', 'I', 'R'])), dtype='int'), index=regions, columns=['S', 'E', 'I', 'R'])
    df_seir['I'] = infects[0, :]
    df_seir['S'] = population - infects[0, :]
    # 存储结果的数组
    preds = []
    opt_betas = [] 
    for i in range(math.ceil(T_train / T_INTERVAL)):
        pred_steps = pred_steps_all[i*T_INTERVAL:(i+1)*T_INTERVAL]
        # 进行PSO参数优化
        cost, pos = run_interval_PSO(pred_steps, df_seir, od_flows, infects)
        if args.show_train:
            print(pred_steps, cost, pos, time.ctime())
        # 使用优化后的参数进行预测
        df_seir, pred = SEIR_Multi(pos, pred_steps, df_seir, od_flows)
        preds.extend(pred)
        opt_betas.append(pos)
    preds = np.array(preds)
    opt_betas = np.array(opt_betas)
    # 保存训练参数结果
    df_seir.to_csv(BASE_SAVE + '/{}_lastseir_train_{}steps.csv'.format(NAME, T_INTERVAL), index=True, header=True)
    np.savetxt(BASE_SAVE + '/{}_betas_train_{}steps.csv'.format(NAME, T_INTERVAL), opt_betas, delimiter=',', fmt='%.8f')
    # 训练集所有的预测结果
    np.savetxt(BASE_SAVE + '/{}_infect_train_{}steps.csv'.format(NAME, T_INTERVAL), preds, delimiter=',', fmt='%d')
    print('final predictions...', preds.shape)
    print('final opt_betas...', opt_betas.shape)
    return preds, opt_betas, df_seir


# 计算误差
def metric(pred, truth):
    RMSE = np.sqrt(np.mean((pred - truth) ** 2))
    MAE = np.mean(np.abs(pred - truth))
    MAPE = np.mean(np.abs(pred - truth) / truth.clip(min=1))
    R2 = 1 - np.mean((pred - truth) ** 2) / np.var(truth)
    return RMSE, MAE, MAPE, R2


# 预测疫情感染人数序列
def predict(od_flows, infects, method):
    assert len(od_flows) == len(infects), 'len of od_flows and infects should be equal'
    T_out = len(od_flows)
    pred_steps_all = np.arange(T_out)
    pred_steps_list = []
    steps_num = math.ceil(T_out / T_INTERVAL)
    for i in range(steps_num-1):
        pred_steps_list.append(pred_steps_all[i*T_INTERVAL:(i+1)*T_INTERVAL])
    pred_steps_list.append(pred_steps_all[(steps_num-1)*T_INTERVAL:])
    # beta参数数据准备
    betas = np.loadtxt(BASE_SAVE + '/{}_betas_train_{}steps.csv'.format(NAME, T_INTERVAL), delimiter=',')
    # 最近的一组beta
    beta = betas[-1, :]
    # 训练集最后，测试集开始的seir矩阵
    df_seir = pd.read_csv(BASE_SAVE + '/{}_lastseir_train_{}steps.csv'.format(NAME, T_INTERVAL), index_col=0)
    # 存储结果的数组
    preds = []
    opt_betas = []
    for pred_steps in pred_steps_list:
        if args.test_opt:
            # 进行PSO参数优化
            cost, pos = run_interval_PSO(pred_steps, df_seir, od_flows, infects)
            if args.show_train:
                print(pred_steps, cost, pos, time.ctime())
            beta = pos
        # 进行多步SEIR预测
        df_seir, pred = SEIR_Multi(beta, pred_steps, df_seir, od_flows)
        preds.extend(pred)
        opt_betas.append(beta)
    preds = np.array(preds)
    opt_betas = np.array(opt_betas)
    # 保存结果
    # 保存参数结果
    df_seir.to_csv(BASE_SAVE + '/{}_lastseir_test_{}steps.csv'.format(NAME, T_INTERVAL), index=True, header=True)
    np.savetxt(BASE_SAVE + '/{}_betas_test_{}steps.csv'.format(NAME, T_INTERVAL), opt_betas, delimiter=',', fmt='%.8f')
    # 测试集所有的预测结果
    np.savetxt(BASE_SAVE + '/{}_infect_{}_{}steps.csv'.format(NAME, method, T_INTERVAL), preds, delimiter=',', fmt='%d')
    print('OD flows, GT infections, predicted infections: ', od_flows.shape, infects.shape, preds.shape)
    # 输出预测误差
    RMSE, MAE, MAPE, R2 = metric(preds, infects)
    print('RMSE: ', RMSE)
    print('MAE: ', MAE)
    print('MAPE: ', MAPE)
    print('R2: ', R2)
    return preds


def train():
    print('---train---')
    OD_TRAIN_PATH = BASE_DATA + '/{}/OD_train_{}.npy'.format(DATASET, DATASET)
    train_odflow = np.load(OD_TRAIN_PATH)
    predictions, opt_betas, df_seir = run_all_PSO(od_flows=train_odflow, infects=train_infection)
    return predictions, opt_betas, df_seir


def test():
    # 测试方法
    # true(真实OD) pred(3DGCformer预测) last(最靠近的结果模拟) avg(前一段时间的平均结果模拟)
    print('---test true OD---')
    OD_TEST_PATH = BASE_DATA + '/{}/OD_true_{}.npy'.format(DATASET, DATASET)
    test_odflow = np.load(OD_TEST_PATH)
    predictions_real = predict(od_flows=test_odflow, infects=test_infection, method='true')

    print('---test pred OD---')
    OD_TEST_PATH = BASE_DATA + '/{}/OD_pred_{}.npy'.format(DATASET, DATASET)
    test_odflow = np.load(OD_TEST_PATH)
    predictions_real = predict(od_flows=test_odflow, infects=test_infection, method='pred')

    print('---test last OD---')
    OD_TEST_PATH = BASE_DATA + '/{}/OD_last_{}.npy'.format(DATASET, DATASET)
    test_odflow = np.load(OD_TEST_PATH)
    predictions_real = predict(od_flows=test_odflow, infects=test_infection, method='last')

    print('---test avg OD---')
    OD_TEST_PATH = BASE_DATA + '/{}/OD_avg_{}.npy'.format(DATASET, DATASET)
    test_odflow = np.load(OD_TEST_PATH)
    predictions_real = predict(od_flows=test_odflow, infects=test_infection, method='avg')


def main():
    train()
    test()


if __name__ == '__main__':
    main()