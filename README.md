# 面向典型突发性风险扰动的城市人口流动性预测方法

## 文件结构
├─README.md // 帮助文档
├─epidemic_pred.py  // 疫情模型的训练和预测过程。
├─OD_pred.py // OD矩阵预测相关模型的实际预测过程。
├─train.py // OD矩阵预测相关模型的训练过程。
│  
├─checkpoint // 存储模型训练过程以及结果文件。
│  ├─checkpoint_epidemic // 存储疫情模型的参数优化结果和预测结果。
│  ├─logs // 存储OD矩阵预测模型的训练日志。
│  └─saves // 存储OD矩阵预测模型的模型文件和训练过程文件。
│  
├─data // 数据集存放的位置
│  ├─ // 与OD矩阵预测模型相关的数据，主要包括区域间OD矩阵、邻接矩阵和时间特征数据。
│  └─data_epidemic // 与疫情模型相关的数据，主要包括区域人口和感染人数数据。
│          
├─models //OD矩阵预测模型3DGCformer的模型文件。
│  ├─ //3DGCformer模型文件以及用于消融实验的一些模型文件。
│  └─Transformer //与Transformer模块相关的组件。
│          
└─utils //小组件。
    ├─command_generate.py // 批量生成服务器上模型的训练指令。
    ├─dataset.py // 基于data中的数据制作用于训练的OD数据集。
    ├─data_process.py // 一些数据处理需要的组件。
    ├─csv_utils.py // 数据集制作的组件。
    └─logger.py // 输出训练日志的组件。

