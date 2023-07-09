import mimetypes
import time

from django.shortcuts import render
from django.views.decorators.http import condition
from rest_framework import viewsets
from rest_framework.response import Response
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from django.http import JsonResponse, HttpResponse, FileResponse
from rest_framework.utils import json
import matplotlib
matplotlib.use('Agg')


def upload1(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            # 执行文件处理操作，例如保存到数据库或其他操作
            # 这里假设你要将文件保存到指定路径下
            file_path = 'static/file/训练集.csv'
            with open(file_path, 'wb') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # 返回成功响应
            response_data = {'message': '文件上传成功'}
            return JsonResponse(response_data)
        else:
            # 返回失败响应，未提供文件
            error_response = {'error': '未提供文件'}
            return JsonResponse(error_response, status=400)
    else:
        # 返回失败响应，无效的请求方法
        error_response = {'error': '无效的请求方法'}
        return JsonResponse(error_response, status=405)

def upload2(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            # 执行文件处理操作，例如保存到数据库或其他操作
            # 这里假设你要将文件保存到指定路径下
            file_path = 'static/file/测试集.csv'
            with open(file_path, 'wb') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # 返回成功响应
            response_data = {'message': '文件上传成功'}
            return JsonResponse(response_data)
        else:
            # 返回失败响应，未提供文件
            error_response = {'error': '未提供文件'}
            return JsonResponse(error_response, status=400)
    else:
        # 返回失败响应，无效的请求方法
        error_response = {'error': '无效的请求方法'}
        return JsonResponse(error_response, status=405)

def upload3(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            # 执行文件处理操作，例如保存到数据库或其他操作
            # 这里假设你要将文件保存到指定路径下
            file_path = 'static/file/验证集.json'
            with open(file_path, 'wb') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # 返回成功响应
            response_data = {'message': '文件上传成功'}
            return JsonResponse(response_data)
        else:
            # 返回失败响应，未提供文件
            error_response = {'error': '未提供文件'}
            return JsonResponse(error_response, status=400)
    else:
        # 返回失败响应，无效的请求方法
        error_response = {'error': '无效的请求方法'}
        return JsonResponse(error_response, status=405)
# 获取参数
def canshu(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))  # 解析 JSON 数据
        k_value = data.get('kValue')  # 获取前端发送的 kValue 数据
        k_value2 = data.get('kValue2')
        # 写入文件
        with open('static/file/data.txt', 'w') as file:
            file.write(f'kValue: {k_value}\n')
            file.write(f'kValue2: {k_value2}\n')

        # 在这里对 kValue 进行后续处理，例如保存到数据库或进行其他操作
        print(k_value)
        print(k_value2)

        # 请求成功的处理逻辑
        response_data = {'message': 'Data received and processed successfully'}
        return JsonResponse(response_data)

        # 请求失败的处理逻辑
    error_response = {'error': 'Invalid request method'}
    return JsonResponse(error_response, status=400)
# 标记Nan值
def Nanfilled(df):
    df = pd.DataFrame(df)

    # 将缺失值替换为NaN
    df = df.replace('None', pd.NaT)
    df = df.replace('NaN', pd.NaT)
    df = df.replace('nan', pd.NaT)
    return df
# knn数据填充
def Knnfilled(df_train, train_index):
    # Create KNNImputer objects
    imputer_train = KNNImputer(n_neighbors=train_index)
    # imputer_valid = KNNImputer(n_neighbors=valid_index)

    # Perform data imputation
    filled_train = imputer_train.fit_transform(df_train)
    # filled_valid = imputer_train.transform(df_valid)  # Use imputer trained on training set

    # Convert the filled data back to DataFrame
    df_train_filled = pd.DataFrame(filled_train, columns=df_train.columns)
    # df_valid_filled = pd.DataFrame(filled_valid, columns=df_valid.columns)

    return df_train_filled

def Knnfilled1(df_train, df_valid,train_index):
    # Create KNNImputer objects
    imputer_train = KNNImputer(n_neighbors=train_index)
    # imputer_valid = KNNImputer(n_neighbors=valid_index)

    # Perform data imputation
    filled_train = imputer_train.fit_transform(df_train)
    filled_valid = imputer_train.transform(df_valid)  # Use imputer trained on training set

    # Convert the filled data back to DataFrame
    # df_train_filled = pd.DataFrame(filled_train, columns=df_train.columns)
    df_valid_filled = pd.DataFrame(filled_valid, columns=df_valid.columns)

    return df_valid_filled

# 计算
def metrics_calculate(pred, y_test, txt_path):
    TP = [0, 0, 0, 0, 0, 0]
    FP = [0, 0, 0, 0, 0, 0]
    FN = [0, 0, 0, 0, 0, 0]
    for i in range(len(y_test)):
        if pred[i] == 0 and y_test[i] == 0:
            TP[0] += 1
        if pred[i] != 0 and y_test[i] == 0:
            FN[0] += 1
        if pred[i] == 0 and y_test[i] != 0:
            FP[0] += 1

        if pred[i] == 1 and y_test[i] == 1:
            TP[1] += 1
        if pred[i] != 1 and y_test[i] == 1:
            FN[1] += 1
        if pred[i] == 1 and y_test[i] != 1:
            FP[1] += 1

        if pred[i] == 2 and y_test[i] == 2:
            TP[2] += 1
        if pred[i] != 2 and y_test[i] == 2:
            FN[2] += 1
        if pred[i] == 2 and y_test[i] != 2:
            FP[2] += 1

        if pred[i] == 3 and y_test[i] == 3:
            TP[3] += 1
        if pred[i] != 3 and y_test[i] == 3:
            FN[3] += 1
        if pred[i] == 3 and y_test[i] != 3:
            FP[3] += 1

        if pred[i] == 4 and y_test[i] == 4:
            TP[4] += 1
        if pred[i] != 4 and y_test[i] == 4:
            FN[4] += 1
        if pred[i] == 4 and y_test[i] != 4:
            FP[4] += 1

        if pred[i] == 5 and y_test[i] == 5:
            TP[5] += 1
        if pred[i] != 5 and y_test[i] == 5:
            FN[5] += 1
        if pred[i] == 5 and y_test[i] != 5:
            FP[5] += 1

    Precision = [0, 0, 0, 0, 0, 0]
    Recall = [0, 0, 0, 0, 0, 0]
    F1 = [0, 0, 0, 0, 0, 0]

    Precision[0] = TP[0] / (TP[0] + FP[0])
    Precision[1] = TP[1] / (TP[1] + FP[1])
    Precision[2] = TP[2] / (TP[2] + FP[2])
    Precision[3] = TP[3] / (TP[3] + FP[3])
    Precision[4] = TP[4] / (TP[4] + FP[4])
    Precision[5] = TP[5] / (TP[5] + FP[5])

    for i in range(6):
        print('Precision'+str(i)+': {}\n'.format(Precision[i]))

    Recall[0] = TP[0] / (TP[0] + FN[0])
    Recall[1] = TP[1] / (TP[1] + FN[1])
    Recall[2] = TP[2] / (TP[2] + FN[2])
    Recall[3] = TP[3] / (TP[3] + FN[3])
    Recall[4] = TP[4] / (TP[4] + FN[4])
    Recall[5] = TP[5] / (TP[5] + FN[5])

    for i in range(6):
        print('Recall'+str(i)+': {}\n'.format(Recall[i]))

    F1[0] = (2 * Precision[0] * Recall[0]) / (Precision[0] + Recall[0])
    F1[1] = (2 * Precision[1] * Recall[1]) / (Precision[1] + Recall[1])
    F1[2] = (2 * Precision[2] * Recall[2]) / (Precision[2] + Recall[2])
    F1[3] = (2 * Precision[3] * Recall[3]) / (Precision[3] + Recall[3])
    if Precision[4] + Recall[4] == 0:
        F1[4] = 0
    else:
        F1[4] = (2 * Precision[4] * Recall[4]) / (Precision[4] + Recall[4])

    # F1[4] = (2 * Precision[4] * Recall[4]) / (Precision[4] + Recall[4])
    F1[5] = (2 * Precision[5] * Recall[5]) / (Precision[5] + Recall[5])

    for i in range(6):
        print('F1('+str(i)+'): {}\n'.format(F1[i]))

    Macro_Precision = sum([Precision[0], Precision[1], Precision[2],
                            Precision[3], Precision[4], Precision[5]]) / 6
    Macro_Recall = sum([Recall[0], Recall[1], Recall[2],
                        Recall[3], Recall[4], Recall[5]]) / 6
    Macro_F1 = sum([F1[0], F1[1], F1[2], F1[3], F1[4], F1[5]]) / 6

    print('Macro_Precision: {}\n'.format(Macro_Precision))

    print('Macro_Recall: {}\n'.format(Macro_Recall))

    print('Macro_F1: {}\n'.format(Macro_F1))


    # l_sum = sum([TP[0], TP[1], TP[2], TP[3], TP[4], TP[5]])
    # m_sum = l_sum + sum([FP[0], FP[1], FP[2], FP[3], FP[4], FP[5]])
    # n_sum = l_sum + sum([FN[0], FN[1], FN[2], FN[3], FN[4], FN[5]])
    #
    # Micro_Precision = l_sum / m_sum
    # print('Micro_Precision: {}\n'.format(Micro_Precision))
    # Micro_Recall = l_sum / n_sum
    # print('Micro_Recall: {}\n'.format(Micro_Recall))
    # Micro_F1 = (2 * Micro_Precision * Micro_Recall) / (Micro_Precision + Micro_Recall)
    # print('Micro_F1: {}\n'.format(Micro_F1))

    f = open(txt_path, 'a', encoding='utf-8')
    for i in range(6):
        f.write('类别{}: '.format(i))
        f.write('\n')
        f.write('Precision: {:.2f}%'.format(Precision[i] * 100))
        f.write('\n')
        f.write('Recall: {:.2f}%'.format(Recall[i] * 100))
        f.write('\n')
        f.write('F1: {:.2f}'.format(F1[i]))
        f.write('\n')
    f.write('Macro_Precision: {:.2f}%'.format(Macro_Precision * 100))
    f.write('\n')
    f.write('Macro_Recall: {:.2f}%'.format(Macro_Recall * 100))
    f.write('\n')
    f.write('Macro_F1: {:.2f}'.format(Macro_F1))
    f.write('\n')
    # f.write('Micro_Precision: {:.2f}%'.format(Micro_Precision * 100))
    # f.write('\n')
    # f.write('Micro_Recall: {:.2f}%'.format(Micro_Recall * 100))
    # f.write('\n')
    # f.write('Micro_F1: {:.2f}'.format(Micro_F1))
    # f.write('\n')
    f.write('\n')
    f.close()

# 检测结果可视化
def check(pred,y_test):
    rows = 6
    cols = 6
    plt.clf()  # 清除当前的图形状态
    array = np.zeros((rows, cols))
    for i in range(len(y_test)):
        array[int(y_test[i])][int(pred[i])] = array[int(y_test[i])][int(pred[i])] + 1
    # for i in range(5):
    #     for j in range(5):
    #         if(array[i][j]==50):
    #             print(i)
    # print(array)
# 使用imshow()函数显示二维数组作为图像
    plt.imshow(array, cmap='viridis')

# 添加文本标签
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, array[i, j], ha='center', va='center', color='w')
    plt.title('COMPARISON OF TEST AND VALIDATION RESULTS', fontdict={'fontsize': 14, 'fontweight': 'bold'})

    # 设置坐标轴范围
    plt.xlabel('PREDICTED LABEL', fontdict={'fontsize': 12, 'color': 'red', 'weight': 'bold'})
    plt.ylabel('TRUE LABEL', fontdict={'fontsize': 12, 'color': 'blue', 'weight': 'bold'})

    # 隐藏坐标轴刻度
    # plt.xticks([])
    # plt.yticks([])

# 显示图像
    plt.savefig('static/file/plot1.png')
# 测试结果可视化
def test_plt(pred):
    pred_flat = pred.ravel()

    # 创建一个包含标签的DataFrame
    df = pd.DataFrame({'labels': pred_flat})

    # 绘制计数图
    sns.countplot(x='labels', data=df)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('VISUALIZATION OF TEST RESULTS')
    plt.savefig('static/file/plot.png')
def verify_plt(pred):
    pred_flat = pred.ravel()

    # 创建一个包含标签的DataFrame
    df = pd.DataFrame({'labels': pred_flat})

    # 绘制计数图
    sns.countplot(x='labels', data=df)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('VISUALIZATION OF VERIFY RESULTS')
    plt.savefig('static/file/plot2.png')
# 按间距中的绿色按钮以运行脚本。
def get_plot2(request):
    file_path =  'static/file/plot2.png'
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='image/png')
    else:
        placeholder_path = 'static/file/placeholder2.png'
        return FileResponse(open(placeholder_path, 'rb'), content_type='image/png')
def get_plot1(request):
    file_path =  'static/file/plot1.png'
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='image/png')
    else:
        placeholder_path = 'static/file/placeholder3.png'
        return FileResponse(open(placeholder_path, 'rb'), content_type='image/png')
def get_plot(request):
    file_path =  'static/file/plot.png'
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='image/png')
    else:
        placeholder_path = 'static/file/placeholder1.png'
        return FileResponse(open(placeholder_path, 'rb'), content_type='image/png')
def start_train(request):
    file_path='static/file/训练集.csv'
    if os.path.exists(file_path):
        df1 = pd.read_csv('static/file/训练集.csv', index_col=None)
        # df2 = pd.read_csv('static/file/测试集.csv', index_col=None)
        # 数据填充
        with open('static/file/data.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.strip().split(': ')
                if key == 'kValue':
                    k_value = int(value)
                elif key == 'kValue2':
                    k_value2 = float(value)
        # 标记Nan
        df1 = Nanfilled(df1)
        # df1=Nanfull0(df1)
        # df2 = Nanfilled(df2)
        # df2=Nanfull0(df2)
        print('开始训练')
        # knn数据填充
        df1 = Knnfilled(df1, k_value)

        # 模型训练
        dtree = DecisionTreeClassifier(
            criterion='entropy',
            min_weight_fraction_leaf=k_value2
        )
        dtree = dtree.fit(df1.iloc[:, 1:-1], df1['label'])

        # 保存模型
        model_path = 'static/file/decision_tree_model.joblib'
        joblib.dump(dtree, model_path)

        print('训练成功')

        response_data = {'message': '训练成功'}
        return JsonResponse(response_data)
    return HttpResponse('File not found or cannot be opened.', status=404)

def start_test(request):
    file_path1='static/file/训练集.csv'
    file_path2='static/file/测试集.csv'
    if os.path.exists(file_path1) and os.path.exists(file_path2):
        df1 = pd.read_csv('static/file/训练集.csv', index_col=None)
        df2 = pd.read_csv('static/file/测试集.csv', index_col=None)
        simple_id = df1['sample_id']
        labels = df1['label']  # 提取标签列
        df1 = df1.drop(['sample_id', 'label'], axis=1)  # 提取特征列

        if 'label' in df2.columns:
            df2 = df2.drop(['sample_id', 'label'], axis=1)
        else:
            df2 = df2.drop(['sample_id'], axis=1)

        # 标记Nan
        df1 = Nanfilled(df1)
        df2 = Nanfilled(df2)
        print('开始测试')
        with open('static/file/data.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.strip().split(': ')
                if key == 'kValue':
                    k_value = int(value)
                elif key == 'kValue2':
                    k_value2 = float(value)
        df2 = Knnfilled1(df1, df2, k_value)

        # 加载模型
        dtree = joblib.load('static/file/decision_tree_model.joblib')

        # 获取测试x，y
        x_test=df2

        # 测试结果
        pred = dtree.predict(x_test)

        # 以字典写入json文件
        pred_dict = {}
        for i, p in enumerate(pred):
            pred_dict[str(i)] = int(p)

        # 将字典保存为JSON文件
        with open('static/file/submit.json', 'w') as f:
            json.dump(pred_dict, f)
        test_plt(pred)

        print('测试成功')
        response_data = {'message': '测试成功'}
        return JsonResponse(response_data)

    return HttpResponse('File not found or cannot be opened.', status=404)
def download_model(request):
    file_path = 'static/file/decision_tree_model.joblib'  # 替换为实际的模型文件路径
    if os.path.exists(file_path):
        response = FileResponse(open(file_path, 'rb'))
        response['Content-Type'] = mimetypes.guess_type(file_path)[0]
        response['Content-Disposition'] = 'attachment; filename=decision_tree_model.joblib'
        return response
    return HttpResponse('File not found or cannot be opened.', status=404)
def download_test(request):
    file_path = 'static/file/submit.json'  # 替换为实际的模型文件路径
    if os.path.exists(file_path):
        response = FileResponse(open(file_path, 'rb'))
        response['Content-Type'] = mimetypes.guess_type(file_path)[0]
        response['Content-Disposition'] = 'attachment; filename=submit.json'
        return response
    return HttpResponse('File not found or cannot be opened.', status=404)

def download_verify(request):
    file_path = 'static/file/软件杯_testresult_RandomForest.txt'  # 替换为实际的模型文件路径
    if os.path.exists(file_path):
        response = FileResponse(open(file_path, 'rb'))
        response['Content-Type'] = mimetypes.guess_type(file_path)[0]
        response['Content-Disposition'] = 'attachment; filename=软件杯_testresult_RandomForest.txt'
        return response
    return HttpResponse('File not found or cannot be opened.', status=404)

def start_verify(request):
    # 获取y_test，并将y_test格式转换
    file_path1='static/file/验证集.json'
    file_path2='static/file/preddata.json'
    if os.path.exists(file_path1) and os.path.exists(file_path2):
        y_test = pd.read_json('static/file/验证集.json', lines=True)
        y_test = np.array(y_test)
        y_test = y_test.flatten()
        y_test1=y_test
        verify_plt(y_test)

        pred = pd.read_json('static/file/submit.json', lines=True)
        pred = np.array(pred)
        pred = pred.flatten()

        y_test_array = y_test
        y_test_reshaped = y_test_array.reshape((-1,))
        txt_path = 'static/file/软件杯_testresult_RandomForest.txt'
        metrics_calculate(pred, y_test_reshaped, txt_path)
        # 检测结果可视化
        check(pred, y_test)

        response_data = {'message': '验证成功'}
        return JsonResponse(response_data)

    return HttpResponse('File not found or cannot be opened.', status=404)

