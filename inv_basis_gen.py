import time
start_time = time.time()

import torch
import sympy as sp
import numpy as np
import random
random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#说明：basis_store中保存所有的基，每个维度的信息为[点类，基序号，矩阵行，矩阵列]
def inv_basis_gen(type_num, matrix_size, basis_num):

    # type_num = 50
    # matrix_size = 32
    # basis_num = 32

    if (basis_num + type_num -1 > matrix_size ** 2 / 2 - matrix_size / 2):
        from warnings import warn
        warn("Number of Basis is too large, the uniqueness of each basis can not be guaranteed", UserWarning)

    # first step
    # Defines the symbolic array of the lower triangular matrix
    a = [sp.symbols(f'a{i}{j}') for i in range(matrix_size) for j in range(0, i + 1)]

    # Converts a symbol array to a lower triangular matrix
    A = sp.Matrix(
        [[a[int(i * (i + 1) / 2 + j)] if j < i else 0 for j in range(matrix_size)] for i in range(matrix_size)]) + sp.eye(
        matrix_size)
    for l in range(matrix_size):
        del a[int(l ** 2 / 2 + 3 * l / 2 - l)]

    elmtr_oprtn_num = matrix_size ** 2
    row_swap_num, row_scale_num, row_add_num = [0] * 3
    # randomly select a type of elementary matrix transformation
    while (not (row_swap_num == elmtr_oprtn_num & row_scale_num == elmtr_oprtn_num & row_add_num == elmtr_oprtn_num)):
        transform = random.choice(['row_swap', 'row_scale', 'row_add'])
        if transform == 'row_swap':
            row1, row2 = random.sample(range(matrix_size), 2)
            A.row_swap(row1, row2)
            row_swap_num += 1
        elif transform == 'row_scale':
            row = random.choice(range(matrix_size))
            coef = random.random()
            A.row_op(row, lambda v, j: (coef * 2 - 1) * v)
            row_scale_num += 1
        elif transform == 'row_add':
            row1, row2 = random.sample(range(3), 2)
            coef = random.random()
            A.row_op(row2, lambda v, j: (coef * 2 - 1) * A[row1, j] + v)
            row_add_num += 1

    # generate orthogonal veectors
    from scipy.stats import ortho_group  # Requires version 0.18 of scipy
    ortho_vec = ortho_group.rvs(dim=int(matrix_size ** 2 / 2 - matrix_size / 2))
    # ortho_vec = torch.tensor(ortho_vec, dtype=torch.float32).to(device)
    # basis_store = np.zeros((type_num, basis_num, matrix_size, matrix_size))
    basis_store = torch.zeros((type_num, basis_num, matrix_size, matrix_size), device=device)

    # generate the first set of basis
    for k in range(basis_num):
        # By default, the number of basis in terms of each type cannot exceed the number of parameters (n**2/2-n/2)
        basis_store[0, k, :, :] = torch.Tensor(A.subs((i, j) for i in a for j in ortho_vec[:, k]).tolist())
        # basis_store[0, k, :, :] = A.subs((i, j) for i in a for j in ortho_vec[:, k])

    # generate the other set of basis
    A0_inv = torch.inverse(basis_store[0, 0, :, :])
    for t in range(type_num - 1):
        if (basis_num + t >= ortho_vec.shape[1]):
            while True:
                basis_store[t + 1, 0, :, :] = np.random.rand(matrix_size, matrix_size)
                rank = np.linalg.matrix_rank(basis_store[t + 1, 0, :, :])
                if rank == matrix_size:
                    break
        else:
            basis_store[t + 1, 0, :, :] = torch.Tensor(A.subs((i, j) for i in a for j in ortho_vec[:, basis_num + t]).tolist())
        intermediate_matrix =  torch.mm(basis_store[t + 1, 0, :, :], A0_inv)
        for k in range(basis_num - 1):
            basis_store[t + 1, k + 1, :, :] = torch.mm(intermediate_matrix, basis_store[0, k + 1, :, :])

    end_time = time.time()
    execution_time = end_time - start_time

    print("execution time is：", execution_time, "s")

    # evalidation
    # print(torch.mm(torch.mm(basis_store[0, 0, :, :], torch.inverse(basis_store[1, 0, :, :])), torch.mm(basis_store[1, 1, :, :], torch.inverse(
    #     basis_store[0, 1, :, :]))))
    # print(torch.mm(torch.mm(basis_store[0, 1, :, :], torch.inverse(basis_store[1, 1, :, :])), torch.mm(basis_store[1, 2, :, :], torch.inverse(
    #     basis_store[0, 2, :, :]))))
    # print(torch.mm(torch.mm(basis_store[2, 1, :, :], torch.inverse(basis_store[1, 1, :, :])), torch.mm(basis_store[1, 2, :, :], torch.inverse(
    #     basis_store[2, 2, :, :]))))
    # print(basis_store[0, 0, :, :] @ np.linalg.inv(basis_store[1, 0, :, :]) @ basis_store[1, 1, :, :] @ np.linalg.inv(
    #     basis_store[0, 1, :, :]))
    # print(basis_store[0, 1, :, :] @ np.linalg.inv(basis_store[1, 1, :, :]) @ basis_store[1, 2, :, :] @ np.linalg.inv(
    #     basis_store[0, 2, :, :]))
    # print(basis_store[2, 1, :, :] @ np.linalg.inv(basis_store[1, 1, :, :]) @ basis_store[1, 2, :, :] @ np.linalg.inv(
    #     basis_store[2, 2, :, :]))
    return(basis_store)
