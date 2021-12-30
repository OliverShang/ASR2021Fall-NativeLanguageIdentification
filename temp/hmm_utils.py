import numpy as np
import pandas as pd
import os
import numpy as np
import torchaudio
from python_speech_features import delta, mfcc, fbank





def logSumBeta(aij_i, mean, var, obs, beta_t1):
    """
    计算beta
    :param aij_i: 转移概率矩阵
    :param mean: 特征均值矩阵
    :param var: 特征方差矩阵
    :param obs: 观测值
    :param beta_t1: 之前时间态下的beta值
    :return: beta对数化后向量的值
    """
    len_x = mean.shape[1]
    y = np.full((len_x,), -np.inf)
    for j in range(0, len_x):
        y[j] = np.log(aij_i[j]) + logGaussian(mean[:, j], var[:, j], obs) + beta_t1[j]
    # 修改matlab的写法
    y_max = np.max(y)
    if y_max == np.inf:
        log_sum_beta = np.inf
    else:
        sum_exp = 0
        for i in range(0, len_x):
            if y_max == -np.inf and y[i] == -np.inf:
                sum_exp = sum_exp + 1
            else:
                sum_exp = sum_exp + np.exp(y[i] - y_max)
        log_sum_beta = y_max + np.log(sum_exp)
    return log_sum_beta


def logSumAlpha(log_alpha_t, aij_j):
    """
    计算alpha
    :param aij_j: 转移概率矩阵
    :param log_alpha_t: 上一个时间状态的log_alpha的值
    :return: alpha对数化后向量的值
    """
    len_x = log_alpha_t.shape[0]
    y = np.full((len_x,), -np.inf)
    for i in range(0, len_x):
        y[i] = log_alpha_t[i] + np.log(aij_j[i])
    # 修改matlab代码
    y_max = np.max(y)
    if y_max == np.inf:
        log_sum_alpha = np.inf
    else:
        sum_exp = 0
        for i in range(0, len_x):
            if y_max == -np.inf and y[i] == -np.inf:
                sum_exp = sum_exp + 1
            else:
                sum_exp = sum_exp + np.exp(y[i] - y_max)
        log_sum_alpha = y_max + np.log(sum_exp)
    return log_sum_alpha


def logGaussian(mean_i, var_i, o_i):
    """
    发射概率
    :param mean_i: 特征均值矩阵
    :param var_i: 方差均值矩阵
    :param o_i: 观测值
    :return: 发射概率
    """
    dim = var_i.shape[0]
    log_b = -1 / 2 * (
            dim * np.log(2 * np.pi) + np.sum(np.log(var_i)) + np.sum((o_i - mean_i) * (o_i - mean_i) / var_i))
    return log_b


def calculate_log_alpha(N, T, aij, mean, var, obs):
    """
    EM_HMM_FR的辅助函数 计算log_alpha
    :param N: 特征数
    :param T: 观测值的时间长度
    :param aij: 概率转移矩阵
    :param mean: 特征均值矩阵
    :param var: 特征方差矩阵
    :param obs: 观测值
    :return: log_alpha
    """
    log_alpha = np.full((N, T + 1), -np.inf)
    # 初始化
    for i in range(0, N):
        log_alpha[i, 0] = np.log(aij[0, i]) + logGaussian(mean[:, i], var[:, i], obs[:, 0])
    # 依照公式计算
    # 由于之前是乘法 对数化后转化为加法
    for t in range(1, T):
        for j in range(1, N - 1):
            log_alpha[j, t] = logSumAlpha(log_alpha[1:N - 1, t - 1], aij[1:N - 1, j]) + \
                              logGaussian(mean[:, j], var[:, j], obs[:, t])
    log_alpha[N - 1, T] = logSumAlpha(log_alpha[1: N - 1, T - 1], aij[1: N - 1, N - 1])
    return log_alpha


def calculate_log_beta(N, T, aij, mean, var, obs):
    """
    EM_HMM_FR的辅助函数 计算log_beta
    :param N: 特征数
    :param T: 观测值的时间长度
    :param aij: 概率转移矩阵
    :param mean: 特征均值矩阵
    :param var: 特征方差矩阵
    :param obs: 观测值
    :return: log_beta
    """
    log_beta = np.full((N, T + 1), -np.inf)
    # 初始化
    log_beta[:, T - 1] = np.log(aij[:, N - 1])
    # 依照公式计算
    # 由于之前是乘法 对数化后转化为加法
    for t in range(T - 2, -1, -1):
        for i in range(1, N - 1):
            log_beta[i, t] = logSumBeta(aij[i, 1:N - 1], mean[:, 1:N - 1], var[:, 1:N - 1], obs[:, t + 1],
                                        log_beta[1:N - 1, t + 1])
    log_beta[N - 1, 0] = logSumBeta(aij[0, 1:N - 1], mean[:, 1: N - 1], var[:, 1: N - 1], obs[:, 0],
                                    log_beta[1: N - 1, 0])
    return log_beta


def calculate_log_Xi(N, T, aij, mean, var, obs, log_alpha, log_beta):
    """
    EM_HMM_FR的辅助函数 计算log_Xi
    :param N: 特征数
    :param T: 观测值的时间长度
    :param aij: 概率转移矩阵
    :param mean: 特征均值矩阵
    :param var: 特征方差矩阵
    :param obs: 观测值
    :param log_alpha: 通过calculate_log_alpha所计算出的值
    :param log_beta: 通过calculate_log_beta所计算出的值
    :return: log_Xi
    """
    log_Xi = np.full((N, N, T), -np.inf)
    for t in range(0, T - 1):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                log_Xi[i, j, t] = log_alpha[i, t] + np.log(aij[i, j]) + \
                                  logGaussian(mean[:, j], var[:, j], obs[:, t + 1]) + log_beta[j, t + 1] - \
                                  log_alpha[N - 1, T]
    for i in range(0, N):
        log_Xi[i, N - 1, T - 1] = log_alpha[i, T - 1] + np.log(aij[i, N - 1]) - log_alpha[N - 1, T]
    return log_Xi


def calculate_gamma(N, T, log_alpha, log_beta):
    """
    EM_HMM_FR的辅助函数 计算gamma
    :param N: 特征数
    :param T: 观测值的时间长度
    :param log_alpha: 通过calculate_log_alpha所计算出的值
    :param log_beta: 通过calculate_log_beta所计算出的值
    :return: gamma
    """
    log_gamma = np.full((N, T), -np.inf)
    for t in range(0, T):
        for i in range(1, N - 1):
            log_gamma[i, t] = log_alpha[i, t] + log_beta[i, t] - log_alpha[N - 1, T]
    gamma = np.exp(log_gamma)
    return gamma


def EM_HMM_FR(mean, var, aij, obs, k):
    """
    模型训练
    :param aij: 概率转移矩阵
    :param mean: 特征均值矩阵
    :param var: 特征方差矩阵
    :param filename: mfcc文件所在路径
    :param k: 并行处理
    :return: mean_numerator 特征均值矩阵的分母
            var_numerator 方差均值矩阵的分母
            aij_numerator 概率转移矩阵的分母
            denominator 特征求和后矩阵的分母
            log_likelihood 最大似然对数值
            likelihood 最大似然值
    """
    dim, T = obs.shape
    mean = np.hstack((np.full((dim, 1), np.NAN), mean, np.full((dim, 1), np.NAN)))
    var = np.hstack((np.full((dim, 1), np.NAN), var, np.full((dim, 1), np.NAN)))
    aij[-1, -1] = 1
    N = mean.shape[1]

    # 准备工作
    # alpha beta Xi gamma
    # 计算log_alpha
    log_alpha = calculate_log_alpha(N, T, aij, mean, var, obs)
    # 计算log_beta
    log_beta = calculate_log_beta(N, T, aij, mean, var, obs)
    # 计算log_Xi
    log_Xi = calculate_log_Xi(N, T, aij, mean, var, obs, log_alpha, log_beta)
    # 计算gamma
    gamma = calculate_gamma(N, T, log_alpha, log_beta)

    # 计算各个矩阵的分子和分母
    mean_numerator = np.zeros((dim, N))
    var_numerator = np.zeros((dim, N))
    denominator = np.zeros((N, 1))
    aij_numerator = np.zeros((N, N))
    for j in range(1, N - 1):
        for t in range(0, T):
            mean_numerator[:, j] = mean_numerator[:, j] + gamma[j, t] * obs[:, t]
            var_numerator[:, j] = var_numerator[:, j] + gamma[j, t] * (obs[:, t] * obs[:, t])
            denominator[j] = denominator[j] + gamma[j, t]
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for t in range(0, T):
                aij_numerator[i, j] = aij_numerator[i, j] + np.exp(log_Xi[i, j, t])
    log_likelihood = log_alpha[N - 1, T]
    likelihood = np.exp(log_alpha[N - 1, T])
    return mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood, k


def viterbi_dist_FR(mean, var, aij, obs):
    """
    viterbi算法
    在原始matlab代码中 引入了时间差
    但实际上时间差分均为1 为简化代码直接将其删除
    :param mean: 特征均值矩阵
    :param var: 特征方差矩阵
    :param aij: 概率转移矩阵
    :param obs: 观测值
    :return: 属于当前类别下的概率
    """
    dim, t_len = obs.shape
    mean = np.hstack((np.full([dim, 1], np.nan), mean, np.full([dim, 1], np.nan)))
    var = np.hstack((np.full([dim, 1], np.nan), var, np.full([dim, 1], np.nan)))
    aij[-1, -1] = 1
    # 扩展一个维度
    if len(aij.shape) == 2:
        aij = aij[:, :, None]
    m_len = mean.shape[1]
    # V矩阵
    fjt = np.full([m_len, t_len], -np.inf)
    # s_chain实际上并没有用到
    # 它记录运算的记录
    # 为保持和matlab代码的一致性 将其保留
    s_chain = [[[] for j in range(0, t_len)] for i in range(0, m_len)]
    for j in range(1, m_len - 1):
        fjt[j, 0] = np.log(aij[0, j, 0]) + logGaussian(mean[:, j], var[:, j], obs[:, 0])
        if fjt[j, 0] > -np.inf:
            s_chain[j][0] = [0, j]
    # viterbi算法主体
    for t in range(1, t_len):
        for j in range(1, m_len - 1):
            f_max = -np.inf
            i_max = -1
            f = -np.inf
            for i in range(1, j + 1):
                if fjt[i, t - 1] > -np.inf:
                    f = fjt[i, t - 1] + np.log(aij[i, j, 0]) + logGaussian(mean[:, j], var[:, j], obs[:, t])
                if f > f_max:
                    f_max = f
                    i_max = i
            if i_max != -1:
                s_chain[j][t].extend(s_chain[i_max][t - 1])
                s_chain[j][t].append(j)
                fjt[j, t] = f_max
    # 选择最优值
    f_opt = -np.inf
    i_opt = -1
    for i in range(1, m_len - 1):
        f = fjt[i, t_len - 1] + np.log(aij[i, m_len - 1, 0])
        if f > f_opt:
            f_opt = f
            i_opt = i
    # 最优值
    if i_opt != -1:
        temp = s_chain[i_opt][t].copy()
        temp.append(m_len)
        chain_opt = temp
    return f_opt
