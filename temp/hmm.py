import os
from hmm_utils import *
from progressbar import *
import concurrent.futures
import numpy as np
def get_data(mode):
    data = []
    data_dict = pd.read_csv(os.path.join("data", "kaggle", mode+".csv"), index_col=0)
    for i in range(len(data_dict)):
        file_path = data_dict.iloc[i, 0]
        waveform, samplerate = torchaudio.load(file_path.replace(".npy", ".mp3"))
        feature = mfcc(
            waveform, samplerate=samplerate, winlen=0.0025, appendEnergy=False
        )
        delta_mfcc = delta(feature, 1)
        delta_delta_mfcc = delta(delta_mfcc, 1)
        mfccs = np.concatenate((feature, delta_mfcc, delta_delta_mfcc), axis=1)
        label = data_dict.iloc[i, 2]
        data.append([label, mfccs.T])  
    return data




class HMM:
    def __init__(self, training_file="training_file.csv", testing_file="testing_file.csv", dim=39,
                 num_of_model=11, num_of_state=12):
        """
        构造函数
        :param training_file: 训练数据csv文件名
        :param testing_file: 测试数据csv文件名
        :param dim: 特征维度
        :param num_of_model: 类别总数
        :param num_of_state: 状态数
        """
        self.training_file = training_file
        self.testing_file = testing_file
        self.dim = dim
        self.num_of_model = num_of_model
        self.num_of_state = num_of_state
        self.mean = np.zeros((dim, num_of_state, num_of_model))
        self.var = np.zeros((dim, num_of_state, num_of_model))
        self.Aij = np.zeros((num_of_state + 2, num_of_state + 2, num_of_model))

    def train_model(self):
        """
        训练模型 num_of_iteration迭代次数
        核心函数在于fit
        :return: 无
        """
        self.EM_initialization_model()
        num_of_iteration = 5
        log_likelihood_iter = [0 for i in range(num_of_iteration)]
        likelihood_iter = [0 for i in range(num_of_iteration)]
        # 进度条
        widgets = ['Training: ', Percentage(), ' ', Bar(), ' ', Timer(), ' ', ETA()]
        train_data = get_data("train")
        progress_bar = ProgressBar(widgets=widgets, maxval=num_of_iteration * len(train_data), term_width=100).start()
        for it in range(0, num_of_iteration):
            log_likelihood, likelihood = self.fit(train_data, it, progress_bar)
            log_likelihood_iter[it] = log_likelihood
            likelihood_iter[it] = likelihood
        np.save("./temp/var.npy", self.var)
        np.save("./temp/Aij.npy", self.Aij)
        np.save("./temp/mean.npy", self.mean)
        progress_bar.finish()

    def fit(self, train_data, it, progress_bar):
        """
        每一轮的训练过程
        :param train_data: 训练数据
        :param it: 迭代次数
        :param progress_bar: 进度条类
        :return: log_likelihood: 最大似然估计对数值
                 likelihood: 最大似然估计值
        """
        # 特征平均值和矩阵分子
        # 特征方差和矩阵分子
        # 概率转移和矩阵分子
        sum_mean_numerator = np.zeros((self.dim, self.num_of_state, self.num_of_model))
        sum_var_numerator = np.zeros((self.dim, self.num_of_state, self.num_of_model))
        sum_aij_numerator = np.zeros((self.num_of_state, self.num_of_state, self.num_of_model))
        sum_denominator = np.zeros((self.num_of_state, self.num_of_model))
        log_likelihood = 0
        likelihood = 0

        mean_by_k = [self.mean[:, :, data[0] - 1] for data in train_data]
        var_by_k = [self.var[:, :, data[0] - 1] for data in train_data]
        aij_by_k = [self.Aij[:, :, data[0] - 1] for data in train_data]
        mfccs = [data[1] for data in train_data]
        k = [data[0] - 1 for data in train_data]
        length = len(k)
        trained = 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for res in executor.map(EM_HMM_FR, mean_by_k, var_by_k, aij_by_k, mfccs, k):
                progress_bar.update(trained + it * length)
                mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood_i, likelihood_i, k = res
                sum_mean_numerator[:, :, k] = sum_mean_numerator[:, :, k] + mean_numerator[:, 1: -1]
                sum_var_numerator[:, :, k] = sum_var_numerator[:, :, k] + var_numerator[:, 1: -1]
                sum_aij_numerator[:, :, k] = sum_aij_numerator[:, :, k] + aij_numerator[1: -1, 1: -1]
                sum_denominator[:, k] = sum_denominator[:, k] + denominator[1: -1].reshape((self.num_of_state,))
                log_likelihood = log_likelihood + log_likelihood_i
                likelihood = likelihood + likelihood_i
                trained += 1

        # 更新特征均值矩阵和特征方差矩阵
        for k in range(0, self.num_of_model):
            for n in range(0, self.num_of_state):
                self.mean[:, n, k] = sum_mean_numerator[:, n, k] / sum_denominator[n, k]
                self.var[:, n, k] = sum_var_numerator[:, n, k] / sum_denominator[n, k] - \
                                    self.mean[:, n, k] * self.mean[:, n, k]
        # 更新概率转移矩阵
        for k in range(0, self.num_of_model):
            for i in range(1, self.num_of_state + 1):
                for j in range(1, self.num_of_state + 1):
                    self.Aij[i, j, k] = sum_aij_numerator[i - 1, j - 1, k] / sum_denominator[i - 1, k]
            self.Aij[self.num_of_state, self.num_of_state + 1, k] = 1 - self.Aij[
                self.num_of_state, self.num_of_state, k]
        self.Aij[self.num_of_state + 1, self.num_of_state + 1, k] = 1
        return log_likelihood, likelihood

    def EM_initialization_model(self):
        """
        计算所有样本的特征和
        计算所有样本的特征平方和
        计算所有样本的特征个数和
        :return: 无
        """
        sum_of_features = np.zeros(self.dim)
        sum_of_features_square = np.zeros(self.dim)
        num_of_feature = 0
        train_data = get_data("train")
        num_of_train = len(train_data)
        for u in range(0, num_of_train):
            feature = train_data[u][1]
            sum_of_features = sum_of_features + np.sum(feature, 1)
            sum_of_features_square = sum_of_features_square + np.sum(feature ** 2, 1)
            num_of_feature = num_of_feature + feature.shape[1]
        self.calculate_initial_EM_HMM_items(sum_of_features, sum_of_features_square, num_of_feature)

    def calculate_initial_EM_HMM_items(self, sum_of_features, sum_of_features_square, num_of_feature):
        """
        隐马尔科夫模型初始化
        概率转移矩阵从当前状态到下一状态设定为0.4
        从当前状态到当前状态设定为0.6
        从初始态到第一个状态设定为1
        :param sum_of_features: 所有样本的特征和
        :param sum_of_features_square: 所有样本的特征平方和
        :param num_of_feature: 特征个数
        :return: 无
        """
        for k in range(0, self.num_of_model):
            for m in range(0, self.num_of_state):
                self.mean[:, m, k] = sum_of_features / num_of_feature
                self.var[:, m, k] = sum_of_features_square / num_of_feature - self.mean[:, m, k] * self.mean[:, m, k]
            for i in range(1, self.num_of_state + 1):
                # 从当前状态到下一状态
                self.Aij[i, i + 1, k] = 0.4
                # 从当前状态到当前状态
                self.Aij[i, i, k] = 1 - self.Aij[i, i + 1, k]
            # 从初始态到第一个状态设定
            self.Aij[0, 1, k] = 1

    def test_model(self):
        # 类别
        num_of_model = 11
        # 错误个数 已完成的测试个数
        num_of_error = 0
        num_of_testing = 0
        test_data = get_data("test")
        # 总测试数据个数
        num_of_test = len(test_data)
        accuracy_history = 0.00
        # 进度条
        widgets = ['Testing:  ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(),
                   ' Acc:{}%'.format(accuracy_history)]
        progress_bar = ProgressBar(widgets=widgets, maxval=num_of_test - 1, term_width=112)
        progress_bar.start()
        # 对每一个测试数据进行校验
        labels = [data[0] - 1 for data in test_data]
        mfccs = [data[1] for data in test_data]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for pre in executor.map(self._test, mfccs, labels):
                if not pre:
                    num_of_error += 1
                progress_bar.widgets[8] = ' Acc: {:.2%}'.format(accuracy_history)
                progress_bar.update(num_of_testing)
                num_of_testing += 1
                accuracy_history = (num_of_testing - num_of_error) / num_of_testing
        progress_bar.finish()
        accuracy_rate = (num_of_testing - num_of_error) * 100 / num_of_testing
        return accuracy_rate

    def _test(self, feature, k):
        """
        并行化的辅助函数
        :param filename: 文件名
        :param k: 类别
        :return:
        """
        f_opt_max = -np.inf
        digit = -1
        # 对每个类别进行检验
        # 计算其概率 并取最高概率
        for p in range(0, self.num_of_model):
            # viterbi算法
            f_opt = viterbi_dist_FR(self.mean[:, :, p], self.var[:, :, p], self.Aij[:, :, p], feature)
            if f_opt > f_opt_max:
                digit = p
                f_opt_max = f_opt
        return digit == k


if __name__ == '__main__':
    np.seterr(all='ignore')
    hmm = HMM()
    hmm.train_model()
    hmm.test_model()