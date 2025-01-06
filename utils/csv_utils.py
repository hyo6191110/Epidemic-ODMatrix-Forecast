from datetime import datetime, timedelta
import csv
import numpy as np
import pandas as pd
import scipy.sparse as ss
import pyarrow.parquet as pq

from azureml.opendatasets import NycTlcGreen
from dateutil import parser


def makeTimestepCSV():
    t = datetime(2021, 10, 1, 0, 0)

    with open("timestep_20211001.csv", "a", encoding="utf-8", newline="") as f:
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        header = ['id', 'start_time', 'end_time']
        csv_writer.writerow(header)
        # 4. 写入csv文件内容
        for i in range(24 * 31):
            end_t = t + timedelta(hours=1)
            row = [i + 1, t, end_t]
            t = end_t
            csv_writer.writerow(row)
        # 5. 关闭文件
        f.close()


def makeNPYFromCSV():
    region_file='csv/GZ_Tianhe_gridlen1000.csv'
    od_file='csv/od_GZ_Tianhe_len1000.csv'
    region_dict={}
    # 1. 获取所有出现的集群区域（区县、grid、或其他类集群）
    i = 0
    with open(region_file, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        # skip header
        next(csv_reader)
        for row in csv_reader:
            # 去除空格
            region_dict[row[0].replace(" ","")]=i
            i=i+1
            print(row[0])
        print(region_dict)
    # 2. 从od整理获取人口流动性数据
    od_list = np.zeros((744, i, i), dtype='float')
    with open(od_file, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        # skip header
        next(csv_reader)
        for row in csv_reader:
            timestep = int(row[0]) - 1
            # 去除空格
            o_region = region_dict[row[1].replace(" ","")]
            d_region = region_dict[row[2].replace(" ","")]
            mobility = row[3]
            od_list[timestep][o_region][d_region] = mobility
    # 3. 保存数据
    np.save("OD_GZ_Tianhe_gridlen1000.npy", od_list)


def makeCSVFromNPY():
    od = np.load("D:\Projects\PycharmProjects\HumanFlow\HumanMobilityPred\datasets\OD_Guangzhou_zone.npy")
    time_slot=od.shape[0]
    num_region=od.shape[1]

    with open("OD_GZZone.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # 先写入columns_name
        col_name_list=['date']
        for i in range(num_region):
            for j in range(num_region):
                col_name="R{}toR{}".format(i, j)
                col_name_list.append(col_name)
        writer.writerow(col_name_list)
        # 表内容
        start_time = datetime(2021, 10, 1, 0, 0)
        for t in range(time_slot):
            row=[start_time]
            start_time = start_time + timedelta(hours=1)
            for i in range(num_region):
                for j in range(num_region):
                    row.append(od[t][i][j])
            writer.writerow(row)
        f.close()


def makeioFlowFromOD():
    od = np.load("D:\Projects\PycharmProjects\HumanFlow\HumanMobilityPred\datasets\OD_Guangzhou_zone.npy")
    time_slot = od.shape[0]
    num_region = od.shape[1]
    oflow = np.array(od.sum(1))
    iflow = np.array(od.sum(2))
    ioflow=np.zeros((2, time_slot, num_region), dtype='float')
    ioflow[0] = iflow
    ioflow[1] = oflow
    ioflow = ioflow.transpose(1, 2, 0)
    print(ioflow.shape)
    np.save("ioFlow_GZZone.npy", ioflow)


# time_slot: time_interval个数
# time_interval： 时间槽 单位为hour 如0.5即为30min
def makeTimeFeatureGZZone(time_slot=744, time_interval=1):
    # 初始值2021/10/01 0:00-0:59
    month_in_year = 10 - 1
    day_in_month = 1 - 1
    day_in_week = 5 - 1
    hour_in_day = 0
    minute_in_hour = 0
    timeF = np.zeros((time_slot, 5), dtype='int')
    for t in range(time_slot):
        timeF[t][0] = month_in_year
        timeF[t][1] = day_in_month + t // 24
        timeF[t][2] = (day_in_week + t // 24) % 7
        timeF[t][3] = hour_in_day + t % 24
        timeF[t][4] = minute_in_hour
    np.save("timeF_GZZone.npy", timeF)
    # check
    for t in range(timeF.shape[0]):
        print("第%d步：%d月%d日星期%d %d:%d" % (t, timeF[t][0]+1, timeF[t][1]+1, timeF[t][2]+1, timeF[t][3], timeF[t][4]))


def makeTimeFeatureJHT(time_slot=1155):
    # 初始值2020/1/1 0:00-23:59
    # 初始值2018/1/1 0:00-23:59
    month_in_year = 1 - 1
    day_in_month = 1 - 1
    day_in_week = 1 - 1
    hour_in_day = 0
    year = 18
    dm=[]
    dm.append([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    dm.append([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    dm.append([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    dm.append([31, 28])
    timeF = np.zeros((time_slot, 5), dtype='int')
    t = 0
    for y in range(len(dm)):
        dm_y=dm[y]
        for m in range(len(dm_y)):
            day=dm_y[m]
            for d in range(day):
                timeF[t][0] = m
                timeF[t][1] = d
                timeF[t][2] = (day_in_week + t) % 7
                timeF[t][3] = 0
                timeF[t][4] = year + y
                t += 1
    np.save("timeF_JHT.npy", timeF)
    # check
    for t in range(timeF.shape[0]):
        print("第%d步：20%d年%d月%d日星期%d" % (t, timeF[t][4], timeF[t][0]+1, timeF[t][1]+1, timeF[t][2]+1))


# 24*4*(30+31)
def makeTimeFeatureNYC(time_slot=5856):
    # 初始值 2013/11/1 0:00-0:15
    # 结束值 2013/12/31 23:45-0:00
    month_in_year = 11 - 1
    day_in_month = 1 - 1
    day_in_week = 5 - 1
    hour_in_day = 0
    minute_in_hour = 0
    dm = [30, 31]
    timeF = np.zeros((time_slot, 5), dtype='int')
    t = 0
    for month in range(len(dm)):
        dm_y = dm[month]
        for day in range(dm_y):
            for hour in range(24):
                for minute in range(4):
                    timeF[t][0] = month_in_year + month
                    timeF[t][1] = day
                    timeF[t][2] = (day_in_week + t // 96) % 7
                    timeF[t][3] = hour
                    timeF[t][4] = minute * 15
                    t += 1
    np.save("timeF_NYC.npy", timeF)
    # check
    for t in range(timeF.shape[0]):
        print("第%d步：%d月%d日星期%d %d:%d" % (t, timeF[t][0] + 1, timeF[t][1] + 1, timeF[t][2] + 1, timeF[t][3], timeF[t][4]))


def makeGZZoneAneighbor():
    A = np.zeros((11, 11))
    mark = [[1, 2, 4, 6],
            [0, 2, 3, 4],
            [0, 1, 3, 5, 6],
            [1, 2, 4, 5],
            [0, 1, 3, 5, 7, 9],
            [2, 3, 4, 6, 9, 10],
            [0, 2, 5, 8],
            [4, 9],
            [6],
            [4, 5, 7, 10],
            [5, 9]]
    for i in range(len(mark)):
        m = mark[i]
        A[i][i] = 1
        for j in range(len(m)):
            A[i][m[j]] = 1
    print(A)
    print(A == A.transpose(1,0))
    np.save("neighbor_GZZone.npy", A)


def makeJHTAneighbor():
    # A_neignbor
    A = np.load('D:\Projects\PycharmProjects\HumanFlow\HumanMobilityPred\datasets\JHT\\adjacency_matrix.npy')
    I = np.eye(A.shape[0])
    A = A + I
    print(A)
    np.save("neighbor_JHT.npy", A)

def makeJHTOD():
    ODPATH ='D:\\大学\\2023研\\2022小组项目\\2.数据材料\\JHT\\od_day20180101_20210228.npz'
    OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2018-01-01', end='2021-02-28', freq='1D')]
    prov_day_data = ss.load_npz(ODPATH)
    prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
    data = prov_day_data_dense[-len(OD_DAYS):, :, :, np.newaxis]
    data=data.reshape(data.shape[0],47,47)
    print(data.shape)
    # log transformation
    ODdata = np.log(data + 1.0)
    np.save("OD_JHT.npy", ODdata)


def logNormData(path="D:\Projects\PycharmProjects\HumanFlow\HMPred-Graph\data\GZZone\ioFlow_GZZone.npy",save='ioFlow_GZZone.npy'):
    od=np.load(path)
    od = np.log(od + 1.0)
    print(od[0])
    np.save(save, od)


def makeODNYC(interval=15):
    secPerTimeslot = int(60 * interval)
    timeslotPerDay = int(1440 / interval)
    timeslot = int(61 * timeslotPerDay)
    start_time = datetime(2013, 11, 1, 0, 0)
    R_dict = {}
    # 1. 获取所有出现的集群区域（区县、grid、或其他类集群）
    i = 0
    with open("D:\\大学\\2023研\\2022小组项目\\2.数据材料\\NYC\\taxi_zone_Manhattan.csv", "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        # skip header
        next(csv_reader)
        for row in csv_reader:
            r=int(row[0].replace(" ", ""))
            # 去除空格
            R_dict[r] = i
            i = i + 1
    ODFlow = np.zeros((timeslot, i, i))
    #od = np.zeros((timeslot, i, i))
    print('num timeslot:',timeslot)
    print('num regions:', i)
    path = ["E:\\Download\\NYC2013\\yellow_tripdata_2013-11.parquet",
            "E:\\Download\\NYC2013\\yellow_tripdata_2013-12.parquet"]
    time_str='tpep_pickup_datetime'
    for c in range(len(path)):
        #if c > 1:
        #    time_str='tpep_pickup_datetime'
        df = pq.read_table(path[c]).to_pandas()
        # 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'PULocationID', 'DOLocationID'
        for index, row in df.iterrows():
            ori = row['PULocationID']
            dest = row['DOLocationID']
            if ori is None or dest is None:
                print('Empty ZoneID')
            num_people = row['passenger_count']
            if num_people > 5:
                print(num_people)
            if ori in R_dict and dest in R_dict and num_people >= 0 and num_people < 10:
                o = R_dict[ori]
                d = R_dict[dest]
                o_time = row[time_str] - start_time
                o_t = o_time.seconds // secPerTimeslot + o_time.days * timeslotPerDay
                ODFlow[o_t][o][d] += num_people + 1
                #od[o_t][o][d] += 1
    #ODFlow = np.log(ODFlow + 1.0)
    # 3. 保存数据
    np.save("OD_NYC.npy", ODFlow)
    #np.save("OD_NYC_1.npy", od)


def makeNYCAneighbor():
    R_dict = {}
    value_to_key={}
    i = 0
    with open("D:\\大学\\2023研\\2022小组项目\\2.数据材料\\NYC\\taxi_zone_Manhattan.csv", "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        # skip header
        next(csv_reader)
        for row in csv_reader:
            r = int(row[0].replace(" ", ""))
            # 去除空格
            R_dict[r] = i
            value_to_key[i]=r
            i = i + 1
    A = np.zeros((i, i))
    with open("D:\\大学\\2023研\\2022小组项目\\2.数据材料\\NYC\\manhattan_adj.csv", "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        # skip header
        next(csv_reader)
        for row in csv_reader:
            region=int(row[0])
            m = R_dict[region]
            adj=row[1].split(';')
            for a in adj:
                if int(a) in R_dict:
                    n=R_dict[int(a)]
                    A[m][n] = 1
                else:
                    print('error')
                    print(region)
    I = np.eye(A.shape[0])
    A = A + I
    # check
    print(A)
    check = A == A.transpose(1, 0)
    for j in range(i):
        if not all(check[j]):
            print('region:',value_to_key[j])
            list=check[j]
            for c in range(i):
                if not list[c]:
                    print(value_to_key[c])
    # 3. 保存数据
    np.save("neighbor_NYC.npy", A)

# 6-26 30-92
def makeOD_WX_EZ(city='WX'):
    if city == 'EZ':
        filenum = 6
        region = 26
        path = "D:\Github\\3DGCformer\data\ezhou\od_get_ezhou ("
        save_path = "OD_EZ.npy"
    else:
        filenum = 30
        region = 92
        path = "D:\Github\\3DGCformer\data\wuxi\od_get_wuxi ("
        save_path = "OD_WX.npy"
    od_list = np.zeros((720, region, region), dtype='int64')

    for i in range(filenum):
        file_path = path + str(i) + ").csv"
        print(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            # skip header
            next(csv_reader)
            for row in csv_reader:
                day = int(row[0]) - 20230601
                hour = int(row[1].split(':')[0])
                timestep = day * 24 + hour
                o_region = int(row[3]) - 1
                d_region = int(row[4]) - 1
                mobility = int(row[5]) - 10
                od_list[timestep][o_region][d_region] += mobility
    # 3. 保存数据
    np.save(save_path, od_list)
    print('end:')
    print(save_path)

def makeTimeFeatureWX_EZ(time_slot=720, time_interval=1):
    # 初始值2023/06/01 0:00-0:59
    month_in_year = 6 - 1
    day_in_month = 1 - 1
    day_in_week = 4 - 1
    hour_in_day = 0
    minute_in_hour = 0
    timeF = np.zeros((time_slot, 5), dtype='int')
    for t in range(time_slot):
        timeF[t][0] = month_in_year
        timeF[t][1] = day_in_month + t // 24
        timeF[t][2] = (day_in_week + t // 24) % 7
        timeF[t][3] = hour_in_day + t % 24
        timeF[t][4] = minute_in_hour
    np.save("timeF_WX.npy", timeF)
    # check
    for t in range(timeF.shape[0]):
        print("第%d步：%d月%d日星期%d %d:%d" % (t, timeF[t][0]+1, timeF[t][1]+1, timeF[t][2]+1, timeF[t][3], timeF[t][4]))

def makeWX_EZAneighbor(city='WX'):
    if city == 'EZ':
        i = 26
        path = "D:\\大学\\2023研\\2022小组项目\\2.数据材料\\DassBI数据\\地图img\\EZ_adj.csv"
        save_path = "neighbor_EZ.npy"
    else:
        i = 92
        path = "D:\\大学\\2023研\\2022小组项目\\2.数据材料\\DassBI数据\\地图img\\WX_adj.csv"
        save_path = "neighbor_WX.npy"
    A = np.zeros((i, i))
    with open(path, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        # skip header
        next(csv_reader)
        for row in csv_reader:
            region = int(row[0])
            m=region-1
            adj=row[1].split(';')
            for a in adj:
                n=int(a)-1
                A[m][n] = 1
    I = np.eye(A.shape[0])
    A = A + I
    # check
    print(A)
    check = A == A.transpose(1, 0)
    for j in range(i):
        if not all(check[j]):
            print('region:',j+1)
            list=check[j]
            for c in range(i):
                if not list[c]:
                    print(c+1)
    # 3. 保存数据
    np.save(save_path, A)

def test1():
    od = np.load("D:\\Projects\\PycharmProjects\\HumanFlow\\其他\\dataset\\NYC\\OD_NYC.npy")
    od = np.exp(od) - 1
    np.save("OD_NYC.npy", od)
    od = np.load("D:\\Projects\\PycharmProjects\\HumanFlow\\其他\\dataset\\NYC\\OD_NYC.npy")
    for i in range(24):
        print(od[i])


def test2():
    od1 = np.load("NYC2014/OD_NYC.npy")
    od = np.load("D:\\Projects\\PycharmProjects\\HumanFlow\\其他\\dataset\\NYC\\OD_NYC.npy")
    od = np.exp(od) - 1
    OD = od + od1
    print(OD.shape)
    np.save("OD_NYC.npy", OD)
    

def test3():
    tf = np.load("D:/Projects/PycharmProjects/HumanFlow/HumanMobilityPred/utils/NYC_30min/timeF_NYC.npy")
    T = int(tf.shape[0] / 2)
    print(T)
    timeF = np.zeros((T, 5), dtype='int')
    for t in range(T):
        timeF[t] = tf[2 * t]
    # check
    for t in range(timeF.shape[0]):
        print("第%d步：%d月%d日星期%d %d:%d" % (
        t, timeF[t][0] + 1, timeF[t][1] + 1, timeF[t][2] + 1, timeF[t][3], timeF[t][4]))
    np.save("timeF_NYC.npy", timeF)


if __name__ == '__main__':
    makeOD_WX_EZ(city='WX')
    makeOD_WX_EZ(city='EZ')






