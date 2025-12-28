import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from kmeans_pytorch import kmeans
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def line_chart(x):
    # 创建目录保存图表
    mkdir('figure')

    # 生成随机数据

    y1 = torch.sin(x)
    y2 = torch.cos(x)
    y3 = torch.sin(x) + torch.cos(x)
    plt.figure()
    # 绘制折线图
    plt.plot(x, y1, 'r-', label='Red Line')
    plt.plot(x, y2, 'y-', label='Yellow Line')
    plt.plot(x, y3, 'b-', label='Blue Line')

    # 设置图表标题和轴标签
    plt.title('Three Lines Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 保存图表
    plt.savefig('figure/three_lines_plot.png')

    plt.close()


def categorie(data):
    # 使用imshow函数绘制热力图
    mkdir('zhuzhuangtu')
    categories =[i for i in range(10)]
    plt.bar(categories, data, color='blue')
    plt.title('Example Bar Chart')
    plt.xlabel('Categories')
    plt.ylabel('Values')    # 显示图形

    plt.savefig('./zhuzhuangtu/'+str(data[0])+'.png')
    plt.close()
def mermory_heat_map(data,id):
    # 使用imshow函数绘制热力图
    mkdir('figure')
    plt.imshow(data, cmap='hot', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    # 显示图形
    plt.savefig('./figure/'+str(id)+'.png')
    plt.close()
def heat_map(data,phase_type,id):
    # 使用imshow函数绘制热力图
    mkdir('quers')
    plt.imshow(data, cmap='hot', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    # 显示图形
    plt.savefig('./quers/'+phase_type+str(id)+'.png')
    plt.close()


def att_mermory_heat_map(data,phase_type,id):
    # 使用imshow函数绘制热力图
    mkdir('scoremermory')
    plt.figure(figsize=(12, 6))
    heatmap = plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto', extent=[0, 100, 0, 10])
    # 添加颜色条
    plt.colorbar(heatmap)
    # 添加边缘线
    # 调整刻度
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 11, 1))
    plt.savefig('./scoremermory/'+phase_type+str(id)+'.png')
    plt.close()
def att_quer_heat_map(data,id):
    # 调整图像大小
    mkdir('scorequery')

    plt.figure(figsize=(12, 6))

    # 画热力图
    heatmap = plt.imshow(data.T, cmap='viridis', interpolation='nearest', aspect='auto', extent=[0, 100, 0, 10])

    # 添加颜色条
    plt.colorbar(heatmap)

    # 添加边缘线
    # 调整刻度
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 11, 1))
    plt.savefig('./scorequery/'+str(id)+'.png')
    plt.close()



def k_means_clustering(x,n_mem,d_model):
    start = time.time()

    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    # sckit-learn xxxx (cuda problem)
    _, cluster_centers = kmeans(X=x, num_clusters=n_mem, distance='euclidean', device=torch.device('cuda:0'))
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers

##1代表异常，0 表示正常
def point_score(outputs, trues):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse(outputs, trues)       #128 * 100* 55
    normal = (1 - torch.exp(-error))   #  128 * 100*55
    score = (torch.sum(normal* loss_func_mse(outputs,trues),dim =-1) / torch.sum(normal,dim=-1))
    return score

def cos_score(outputs, trues):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse(outputs, trues)       #128 * 100* 55
    error = F.softmax(error,dim=-1)
    normal = 1/(1-error)   # 128 * 100
    score = (torch.sum(normal* loss_func_mse(outputs,trues),dim =-1) / torch.sum(normal,dim=-1))
    return score


def visualization(dataset,best_thresh,point_label,anomaly_score,true_labels,true_list):
    win_size = 200
    path_true = os.path.join(os.getcwd(), "visu", dataset, "true")
    path_score = os.path.join(os.getcwd(), "visu", dataset, "Score")
    if (not os.path.exists(path_true)):
        mkdir(path_true)
    if (not os.path.exists(path_score)):
        mkdir(path_score)
    index_len =(point_label.shape[0] - win_size) // win_size+ 1
    for i in range(0,index_len):
        index = i * win_size
        # print(index)
        pre_label = point_label[index:index+win_size]
        score =anomaly_score[index:index+win_size]
        true_label =true_labels[index:index+win_size]
        true =true_list[index:index+win_size]


        plt.figure()
        plt.tick_params()
        plt.plot(true, label='Ground truth')
        Anomaly =True
        for i, value in enumerate(true_label):
            if value == 1:
                if Anomaly:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3,ymin=0.5, ymax=1, label='True anomaly')
                    Anomaly = False
                else:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3,ymin=0.5, ymax=1)

        Anomaly1 =True
        for i, value in enumerate(pre_label):
            if value == 1:
                if Anomaly1:
                    plt.axvspan(i, i + 1, color='green', alpha=0.3,ymin=0.1, ymax=0.4, label='pre anomaly')
                    Anomaly1 = False
                else:
                    plt.axvspan(i, i + 1, color='green', alpha=0.3,ymin=0.1, ymax=0.4 )


        plt.legend()
        plt.xlabel('Time points')
        x_major_locator = MultipleLocator(50)
        # 把x轴的刻度间隔设置为1，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        plt.savefig(path_true+'/'+str(index)+'.png')
        plt.close()


        plt.figure()
        plt.plot(score, label='Anomaly score')
        plt.tick_params()
        plt.axhline(y=best_thresh, color='red', linestyle='--', label='Threshold')
        Anomaly =True
        for i, value in enumerate(pre_label):
            if value == 1:
                if Anomaly:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3, label='Pred anomaly')
                    Anomaly = False
                else:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3)
        plt.legend()
        plt.xlabel('Time points')
        x_major_locator = MultipleLocator(50)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.savefig(path_score+'/'+str(index)+'.png')
        plt.close()