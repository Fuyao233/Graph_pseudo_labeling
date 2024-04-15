import numpy as np
import copy
from scipy.stats import ortho_group

#basis_dim 基矩阵的尺寸
#class_num 类数量
#basis_num 基数量
#h 对角线加的值

def basis_process_dim32_class2(basis_dim: int = 32, basis_num: int = 4, h: float = 0.01) -> np:
    # Generate Involutory Matrix list
    identity_matrix = np.eye(basis_dim)
    involutory_list = []
    # involutory_list = np.zeros((int(basis_dim * (basis_dim - 1) / 2), basis_dim, basis_dim))
    for i in range(1, basis_dim - 1):
        for j in range(i):
            tmp = copy.deepcopy(identity_matrix)
            tmp[[i, j]] = tmp[[j, i]]
            # involutory_list[int((i - 1) * i / 2 + j), :, :] = tmp
            involutory_list.append(tmp)
    del tmp

    # round 1 AB
    s1 = involutory_list[1] - involutory_list[0]
    s2 = involutory_list[1] - np.eye(basis_dim)

    YAB = np.random.randn(basis_dim, basis_dim)
    A2B2inv = (np.diag([1 + h] * basis_dim) - np.linalg.pinv(s1) @ s1) @ YAB @ (
                    np.diag([1 + h] * basis_dim) - s2 @ np.linalg.pinv(s2))
    A1B1inv = involutory_list[0] @ A2B2inv
    ortho_vec = ortho_group.rvs(dim=basis_dim**2)
    ortho_index = 0
    basis = np.zeros([2, basis_num, basis_dim, basis_dim])

    for i in range(int(basis_num / 2)):
        basis[1, 2 * i, :, :] = ortho_vec[ortho_index].reshape(basis_dim, basis_dim)
        ortho_index += 1
        basis[0, 2 * i, :, :] = A1B1inv @ basis[1, 2 * i, :, :]
        basis[1, 2 * i + 1, :, :] = ortho_vec[ortho_index].reshape(basis_dim, basis_dim)
        ortho_index += 1
        basis[0, 2 * i + 1, :, :] = A2B2inv @ basis[1, 2 * i + 1, :, :]
    return basis

if __name__ == '__main__':
    m=1
    n=1/2
    y = 1/(-m**2/n+n)
    x = -m*y/n

    basis_num = 8
    np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
    basis = basis_process_dim32_class2(basis_num = basis_num)
    for i in range(int(basis_num/2)):
        print(
            np.array2string(
                (x * basis[0, 2 * i, :, :] @ np.linalg.inv(basis[1, 2 * i, :, :]) + y * basis[0, 2 * i + 1, :,
                                                                                        :] @ np.linalg.inv(
                    basis[1, 2 * i + 1, :, :]))
                @ (m * basis[1, 2 * i, :, :] @ np.linalg.inv(basis[0, 2 * i, :, :]) + n * basis[1, 2 * i + 1, :,
                                                                                          :] @ np.linalg.inv(
                    basis[0, 2 * i + 1, :, :])), separator=', '))
