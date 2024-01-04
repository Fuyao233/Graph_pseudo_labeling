import sympy as sp
import numpy as np
import random

random.seed(1)
from scipy import linalg

type_num = 3
matrix_size = 32
basis_num = 32  # should be less than matrix_size

# first step
# 定义下三角矩阵的符号数组
a = [sp.symbols(f'a{i}{j}') for i in range(matrix_size) for j in range(0, i+1)]
b = [sp.symbols(f'b{i}{j}') for i in range(matrix_size) for j in range(0, i+1)]

# 将符号数组转换为下三角矩阵
A = sp.Matrix([[a[int(i * (i + 1) / 2 + j)] if j <= i else 0 for j in range(matrix_size)] for i in range(matrix_size)])
B = sp.Matrix([[b[int(i * (i + 1) / 2 + j)] if j <= i else 0 for j in range(matrix_size)] for i in range(matrix_size)])

#问题1：变换次数太少的话会报错，太多没事。
#解答1：变换次数太少AB太简单，导致AB=M直接有平凡常数等式（0=0），相当于将某些等式可以直接忽视，设置range(0)就能看出来
#解决1：设置自动变化的初等变换次数，需要实验来确定等式

elmtr_oprtn_num = 1000
# 随机选择一种矩阵初等变换
for matrix in [A, B]:
    for i in range(elmtr_oprtn_num):
        transform = random.choice(['row_swap', 'row_scale', 'row_add'])
        # 执行随机的矩阵初等变换
        if transform == 'row_swap':
            # 随机选择两行进行交换
            row1, row2 = random.sample(range(matrix_size), 2)
            matrix.row_swap(row1, row2)
        elif transform == 'row_scale':
            # 随机选择一行和一个非零标量进行缩放
            row = random.choice(range(matrix_size))
            coef = random.random()
            matrix.row_op(row, lambda v, j: (coef * 2 - 1) * v)
        elif transform == 'row_add':
            # 随机选择两行，其中一行乘以一个标量后与另一行相加
            row1, row2 = random.sample(range(3), 2)
            coef = random.random()
            matrix.row_op(row2, lambda v, j: (coef * 2 - 1) * matrix[row1, j] + v)

# 打印变换后的矩阵
# print(A * B)

#问题2：对角线上的参数如果为0那么会不满秩（singular）
#解决2：恰好有n个自由度，可以直接设定对角线上的值为非零

# A = A + sp.eye(matrix_size)
# B = B + sp.eye(matrix_size)

# 生成随机矩阵
M = sp.eye(matrix_size)
# M = np.random.rand(n, n)
# # 检查矩阵的秩
# rank = np.linalg.matrix_rank(M)
# # 如果矩阵不是满秩，则重新生成
# while rank != n:
#     M = np.random.rand(n, n)
#     rank = np.linalg.matrix_rank(M)
# print(M)

# 线性化

AB = (A * B)
expr = []
for i in range(matrix_size):
    for j in range(matrix_size):
        expr += [sp.Eq(AB[i, j], M[i, j])]
# exp=sp.Eq(i for i in AB, j for j in M_flat)

from scipy.stats import ortho_group  # Requires version 0.18 of scipy

ortho_vec = ortho_group.rvs(dim=matrix_size)
ortho_vec = ortho_vec/(100*np.min(ortho_vec))  #in case the numbers are too small

A_store = np.zeros((basis_num, matrix_size, matrix_size))
B_store = np.zeros((basis_num, matrix_size, matrix_size))

for num in range(basis_num):
    # 方案一：直接带入（必须要是原始下三角矩阵的对角线元素才行）
    expr_t = expr
    a_t = a.copy()
    expr_len = len(expr_t)
    for l in range(matrix_size):
        for k in range(expr_len):
            expr_t[k] = expr_t[k].subs(a_t[int(l**2/2+3*l/2-l)], ortho_vec[l, num])
        del a_t[int(l**2/2+3*l/2-l)]
            # expr_t[k] = expr_t[k].subs((i, j) for i in a[0:matrix_size] for j in ortho_vec[:, num])
    result = sp.nsolve(expr_t, a_t + b, [random.random() for i in range(len(a) + len(b) - matrix_size)],
                       tol=1e-3)
    # result = sp.nsolve(expr_t, a[matrix_size:] + b, [random.random() for i in range(len(a) + len(b) - matrix_size)],
    #                    tol=1e-3)
    a_t.remove()
    # 方案二：加条件
    # expr_t = expr
    # for k in range(matrix_size):
    #     expr += [sp.Eq(a[k], ortho_vec[k, num])]
    # result = sp.nsolve(expr_t, a + b,
    #                    ortho_vec[:, num].tolist() + [random.random() for i in range(len(a) + len(b) - matrix_size)],
    #                    tol=1e-3)

    result_full = (ortho_vec[:, num].tolist() + np.ravel(result).tolist())
    A_t = B_t = np.zeros_like(AB).astype(float)
    for i in range(AB.shape[0]):
        for j in range(AB.shape[1]):
            A_t[i, j] = A[i, j].subs((ii, jj) for ii in a + b for jj in result_full)
            B_t[i, j] = B[i, j].subs((ii, jj) for ii in a + b for jj in result_full)
    B_t = linalg.inv(B_t)
    A_store[num, :, :], B_store[num, :, :] = [A_t, B_t]
    # evalidation
    # AB_t = sp.zeros(AB.shape[0],AB.shape[1])
    # for i in range(AB.shape[0]):
    #     for j in range(AB.shape[1]):
    #         AB_t[i,j] = AB[i,j].subs((ii,jj) for ii in a+b for jj in result_full)
    # print(AB_t)

# while(flag&count<10):
#     try:
#         result=sp.nsolve(expr_t,a[n:]+b,[random.random() for i in range(len(a)+len(b)-n)],tol=1e-6)
#     except Exception:
#         count += 1
#         continue
# print(result)
# exp_final = exp_t.subs((i, j) for i in a[3:]+b for j in result)
# print(exp_final)
