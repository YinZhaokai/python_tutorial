#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   task2.py
@Time    :   2020/02/04 14:27:38
@Author  :   Zhaokai Yin
@Version :   0.1
@Contact :   yinzhaokai@foxmail.com
@License :   (C)Copyright 2017-2020
'''

# here put the import lib
import numpy as np

#使用python内置功能读写数据文件并求矩阵转置
def func1(inputfile):
    """[使用python内置功能读写数据文件并求矩阵转置]
    
    Arguments:
        inputfile {[str]} -- [输入的文件名称]
    """
    #读取数据文件
    #注意，使用windows记事本保存的文件编码格式为 UTF-8 BOM，与UTF-8不同，需要转换编码格式
    with open(inputfile, 'r', encoding = 'utf-8') as f:
        #f.readlines()--读取多行，返回的结果为list类型
        #但读取后每行结尾会有换行符'\n'
        # context = f.readlines()

        #f.read()--将文件中的所有内容视作完整的字符串读取进来,包含可能的制表符'\t'、换行符'\n'等
        context2 = f.read()    
    #将读取的文本按照换行符'\n'分割为多行，lines变量为list类型，每个元素为每行的内容（字符串类型）
    lines = context2.split('\n')

    matrix = []  #定义新的list存放文件中的矩阵
    #遍历每行
    for line in lines:
        #将每行数据按照' '分割为多个数字文本构成的list（数字仍然是字符串类型）
        numbers = line.split(' ')        
        #将一行的数据保存在矩阵中(此时数字仍然是字符串类型)
        matrix.append(numbers)
    new_matrix = []  #定义新的list存放转置后的矩阵    

    #按照先列（i）后行（j）的顺序遍历matrix(即顺序为11,21,31,12,22,31,13,23,33)
    for j in range(len(matrix[0])):
        new_line = []   #定义新的list存放转之后矩阵的一行
        for i in range(len(matrix)):
            print(matrix[i][j])
            new_line.append(matrix[i][j])
        #将新的一行存入新的矩阵中
        new_matrix.append(new_line)

    #将转置后的矩阵写入文件中
    with open('task2_out_1.txt', 'w', encoding='utf-8') as f:
        #遍历新矩阵的每行
        for i in range(len(new_matrix)):
            #将矩阵中每行元素用空格连起来，形成字符串
            line_str = ' '.join(new_matrix[i])
            #写完一行要记得加换行符
            f.write(line_str + '\n')
    return

#使用numpy工具包读写数据文件并求矩阵转置
def func2(inputfile):
    """[使用numpy工具包读写数据文件并求矩阵转置]
    
    Arguments:
        inputfile {[str]} -- [输入的文件名称]
    """
    #使用numpy包自带的读取文件函数读取数据文件并保存为二维矩阵
    matrix = np.loadtxt(inputfile, dtype='int32', encoding='utf-8')
    #使用numpy包自带的矩阵转置函数求矩阵的转置
    new_matrix = matrix.T
    #使用numpy包自带的写入文件函数将转之后的矩阵写入文件中
    np.savetxt('task2_out_2.txt',new_matrix,fmt='%s',delimiter=' ',encoding='utf-8')
    
    return


def main():
    # func1('task2_in.txt')
    func2('task2_in.txt')
    return
if __name__=='__main__':
    main()
