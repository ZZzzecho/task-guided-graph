import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
import os
from scipy.stats import spearmanr

import numpy as np
from scipy.stats import rankdata
from scipy.stats import spearmanr
from multiprocessing import Pool, RawArray
from math import sin, pi
import multiprocessing
from functools import partial

y=[]
for i in range(50):
    embedding_df = pd.read_csv(f"/laijizheng/QwenGAT/phykeyword_embeddings/64_version_{i+1}_embeddings.csv")
    keywords = embedding_df.iloc[:, 0].tolist()  # 第一列为关键词名称
    embeddings_array = embedding_df.iloc[:, 1:].values.astype(np.float32)
    y.append(embeddings_array)

y=np.array(y).transpose(1, 2, 0)
p, q, n = y.shape
from scipy.stats import spearmanr
from sklearn.covariance import GraphicalLasso
from joblib import Parallel, delayed
from tqdm import tqdm
import numba
from sklearn.covariance import LedoitWolf
import numpy as np
from numba import cuda, float32
from numba import njit
import math


# 设备函数：计算单个特征的秩次（GPU核心）
@cuda.jit(device=True)
def compute_rank(column, rank_out):
    n = column.size
    # 生成索引并排序
    idx = cuda.local.array(shape=1024, dtype=numba.int32)  # 假设最大样本数1024
    for i in range(n):
        idx[i] = i
    # 冒泡排序（小规模数据高效）
    for i in range(n - 1):
        for j in range(n - i - 1):
            if column[idx[j]] > column[idx[j + 1]]:
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
    # 计算秩次（处理并列情况）
    for i in range(n):
        count = 1
        sum_rank = i
        # 检查相同值
        j = i + 1
        while j < n and column[idx[j]] == column[idx[i]]:
            sum_rank += j
            count += 1
            j += 1
        avg_rank = sum_rank / count + 1  # +1因秩次从1开始
        for k in range(i, i + count):
            rank_out[idx[k]] = avg_rank
        i += count - 1  # 跳过已处理值


# 内核函数：并行计算特征对的相关性
@cuda.jit
def spearman_kernel(vecy, spearman_matrix):
    i, j = cuda.grid(2)  # 2D网格
    n_features, n_samples = vecy.shape

    if i < n_features and j < n_features and i <= j:
        # 动态分配共享内存存储秩次
        ri = cuda.shared.array(shape=(1024,), dtype=numba.float32)  # 每线程块共享
        rj = cuda.shared.array(shape=(1024,), dtype=numba.float32)

        # 计算当前特征对的秩次
        compute_rank(vecy[i], ri)
        compute_rank(vecy[j], rj)

        # 同步线程确保秩次计算完成
        cuda.syncthreads()

        # 计算均值
        mean_i = 0.0
        mean_j = 0.0
        for k in range(n_samples):
            mean_i += ri[k]
            mean_j += rj[k]
        mean_i /= n_samples
        mean_j /= n_samples

        # 计算协方差和标准差
        cov = 0.0
        std_i = 0.0
        std_j = 0.0
        for k in range(n_samples):
            dev_i = ri[k] - mean_i
            dev_j = rj[k] - mean_j
            cov += dev_i * dev_j
            std_i += dev_i ** 2
            std_j += dev_j ** 2

        # 写入结果（对称矩阵）
        corr = cov / (math.sqrt(std_i) * math.sqrt(std_j))
        spearman_matrix[i, j] = corr
        spearman_matrix[j, i] = corr


# 主机函数：调用GPU计算
def gpu_spearman(vecy):
    n_features, n_samples = vecy.shape
    d_vecy = cuda.to_device(vecy.astype(np.float32))  # 传输数据到设备
    d_matrix = cuda.device_array((n_features, n_features), dtype=np.float32)

    # 配置2D网格（16x16线程块）
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (n_features + 15) // 16,
        (n_features + 15) // 16
    )

    # 启动内核
    spearman_kernel[blocks_per_grid, threads_per_block](d_vecy, d_matrix)

    # 取回结果
    return d_matrix.copy_to_host()


# ================== 预计算优化层 ==================
@numba.jit(nopython=True, parallel=True)
def precompute_spearman(vecy):
    """预计算所有特征对的Spearman相关系数"""
    n_features, n_samples = vecy.shape
    spearman_matrix = np.zeros((n_features, n_features))
    for i in numba.prange(n_features):
        for j in numba.prange(i, n_features):
            ri = np.argsort(vecy[i]).argsort()
            rj = np.argsort(vecy[j]).argsort()
            cov = np.cov(ri, rj)
            spearman_matrix[i, j] = cov[0, 1] / (np.std(ri) * np.std(rj))
            spearman_matrix[j, i] = spearman_matrix[i, j]
    return spearman_matrix


# ================== 矩阵运算优化层 ==================
def KRK_optimized(l, m, spearman_matrix, p, q):
    """向量化实现KRK计算"""
    rows = np.arange(m - 1, (q - 1) * p + m, p)
    cols = np.arange(l - 1, (q - 1) * p + l, p)
    sub_matrix = spearman_matrix[np.ix_(rows, cols)]
    return 2 * np.sin(np.pi / 6 * sub_matrix)


def LRL_optimized(l, m, spearman_matrix, p, q):
    """向量化实现LRL计算"""
    rows = np.arange((l - 1) * p, l * p)
    cols = np.arange((m - 1) * p, m * p)
    sub_matrix = spearman_matrix[np.ix_(rows, cols)]
    return 2 * np.sin(np.pi / 6 * sub_matrix)


# ================== 并行计算优化层 ==================
def compute_R_matrix_parallel(mode, matrix, spearman_matrix, p, q, n_jobs=-1):
    """通用并行计算R矩阵"""
    size = p if mode == 'B' else q
    func = KRK_optimized if mode == 'B' else LRL_optimized
    def process_block(i):
        row = np.zeros(size, dtype=np.float32)
        for j in range(size):
            block = func(i + 1, j + 1, spearman_matrix, p, q)  # (p x p) 或 (q x q)
            # 用 float32 可减小内存 & 足够
            row[j] = np.trace(block @ matrix).astype(np.float32)
        return row
    # 线程后端 + 禁用 memmap，避免写临时盘
    results = Parallel(
        n_jobs=n_jobs,
        backend="threading",
        prefer="threads",
        max_nbytes=None          # 关键：不做内存映射到磁盘
    )(delayed(process_block)(i) for i in range(size))
    return np.asarray(results, dtype=np.float32)



from scipy.linalg import svd


def stabilize_matrix(matrix, min_eig=1e-4):
    """通过SVD截断稳定病态矩阵"""
    U, s, Vh = svd(matrix)
    s[s < min_eig] = min_eig  # 截断小奇异值
    return U @ np.diag(s) @ Vh


def MNglasso_optimized(y, lam_A, lam_B, max_iter=8, n_jobs=-1):
    p, q, n = y.shape
    # spearman_matrix = precompute_spearman(y.reshape(p*q, n))
    spearman_matrix = gpu_spearman(y.reshape(p * q, n))
    #np.save(f'/laijizheng/spearman_mat/phy_sp_{p}_{q}.npy', spearman_matrix)
    # spearman_matrix =np.load(f'/laijizheng/spearman_mat/phy_sp_{p}_{q}.npy')
    # 初始化精度矩阵
    A = np.eye(p)
    B = np.eye(q)

    for iter in range(max_iter):
        # 更新A矩阵 =================================
        SA = compute_R_matrix_parallel('B', B, spearman_matrix, p, q, n_jobs)
        SA = (SA - SA.min()) / (SA.max() - SA.min())
        # 增强的GraphicalLasso参数
        try:
            model_A = GraphicalLasso(
                alpha=lam_A / q,
                mode='cd',
                tol=1e-5,  # 降低收敛阈值
                enet_tol=1e-6,  # 增加弹性网容忍度
                max_iter=100  # 增加迭代次数
            ).fit(SA)
        except:
            SA = SA + 0.01 * np.eye(SA.shape[0])
            model_A = GraphicalLasso(
                alpha=lam_A / q,
                mode='cd',
                tol=1e-5,  # 降低收敛阈值
                enet_tol=1e-6,  # 增加弹性网容忍度
                max_iter=100  # 增加迭代次数
            ).fit(SA)
        wiA = model_A.precision_
        rel = {}
        for i in range(p):
            for j in range(i + 1, p):
                if wiA[i, j] != 0:
                    rel[keywords[i] + '--' + keywords[j]] = abs(wiA[i, j])
        print(f'number of edges: {len(rel)}.')
        np.save(f'/laijizheng/QwenGAT/A_phy/A_{lam_A}_{iter}_{len(rel)}.npy', wiA)
        # 更新B矩阵 =================================
        SB = compute_R_matrix_parallel('A', wiA, spearman_matrix, p, q, n_jobs)
        SB = (SB - SB.min()) / (SB.max() - SB.min())

        # # 增强的正则化方案
        # SB_reg = LedoitWolf().fit(SB).covariance_
        # SB_reg = stabilize_matrix(SAB_reg)
        # cond_num = np.linalg.cond(SB_reg)
        # reg_strength = max(1e-4, 1/(cond_num/1e8))
        # SB_reg = SB_reg + reg_strength * np.eye(SB_reg.shape[0])

        # print(f"Iter {iter+1}-B: Cond={cond_num:.2e}, Reg={reg_strength:.1e}")
        try:
            model_B = GraphicalLasso(
                alpha=lam_B / p,
                mode='cd',
                tol=1e-5,
                enet_tol=1e-6,
                max_iter=100
            ).fit(SB)
            wiB = model_B.precision_
        except:
            SB = SB + 0.01 * np.eye(SB.shape[0])
            model_B = GraphicalLasso(
                alpha=lam_B / p,
                mode='cd',
                tol=1e-5,
                enet_tol=1e-6,
                max_iter=100
            ).fit(SB)
            wiB = model_B.precision_

        # 收敛判断
        diff = np.linalg.norm(wiA - A) + np.linalg.norm(wiB - B)
        print(f"Iter {iter + 1}: Diff={diff:.4f}")
        A, B = wiA.copy(), wiB.copy()

    # 标准化
    fac = B[0, 0]
    return A, (1 / fac) * B


import warnings
warnings.filterwarnings('ignore')
# 执行优化算法
lam_A=2.8
#lam_B=float(np.sqrt(((n*lam_A**2-np.log(4))*q/p+np.log(4))/n))
lam_B=32
print(lam_B)
A_est, B_est = MNglasso_optimized(y, lam_A=lam_A, lam_B=10*lam_A, n_jobs=4)

# import warnings
# warnings.filterwarnings('ignore')
# for i in range(2):
#     lam_A=2.4+i*0.1
#     A_est, B_est = MNglasso_optimized(y, lam_A=lam_A, lam_B=10*lam_A,max_iter=8, n_jobs=12)
