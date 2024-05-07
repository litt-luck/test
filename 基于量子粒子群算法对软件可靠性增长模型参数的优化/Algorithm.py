import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error


# 构建神经网络模型
def build_model(params, X):
    """
    构建神经网络模型
    :param params:神经网络模型的参数    X: 输入特征矩阵        model: 构建好的神经网络模型
    :return:model 训练好的模型
    """
    model = Sequential()
    model.add(Dense(params[0], input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# 定义目标函数（适应度函数）
def objective_func(params, X, y):
    """
    目标函数（适应度函数）
    :param params: 神经网络模型的参数        X: 输入特征矩阵        y: 目标标签
    :return: mse: 均方误差作为目标函数值
    """
    # 根据参数构建神经网络模型
    model = build_model(params, X)

    # 使用训练数据进行模型训练
    model.fit(X, y, epochs=10, verbose=0)

    # 在训练数据上进行预测
    y_pred = model.predict(X)

    # 计算均方误差作为目标函数值
    mse = mean_squared_error(y, y_pred)

    return mse


def data_handle(file_path):
    """
    读取数据并进行预处理，提取特征和目标标签
    :param file_path: 文件的绝对路径
    :return: 特征列表和目标标签列表
    """
    with open(file_path, encoding="utf-8") as f:
        header = []  # 存储数据集的特征列名
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])  # 提取每行以"@attribute"开头的特征列名
            elif line.startswith("@data"):
                break  # 遇到"@data"行后停止读取文件头部信息
        df = pd.read_csv(f, header=None)  # 使用 pandas 读取文件剩余内容作为数据集
        df.columns = header  # 将特征列名赋值给数据集的列名
    list_feature = df.iloc[:, :-1].values.tolist()  # 提取数据集中除最后一列外的所有特征列，并将其转换为列表
    category_labels = [0 if label == 'N' else 1 for label in df.iloc[:, -1]]  # 将数据集最后一列的标签进行二分类转换，'N'转为0，其他值转为1
    return list_feature, category_labels


# 定义QPSO类
class QPSO:
    def __init__(self, params, num_particles=30, num_dimensions=10, max_iterations=100, lb=2, ub=50):
        self.num_particles = num_particles  # 粒子数量
        self.num_dimensions = num_dimensions  # 维度数量
        self.max_iterations = max_iterations  # 最大迭代次数
        self.lb = lb  # 搜索空间下界
        self.ub = ub  # 搜索空间上界
        self.objective_func_params = params  # 目标函数参数

        self.global_best_position = None  # 全局最优位置
        self.global_best_fitness = np.inf  # 全局最优适应度

        self.particles_position = None  # 粒子位置
        self.particles_velocity = None  # 粒子速度
        self.particles_best_position = None  # 粒子历史最优位置
        self.particles_best_fitness = np.inf  # 粒子历史最优适应度

    def initialize_particles(self):
        # 初始化粒子位置和速度
        self.particles_position = np.random.uniform(low=self.lb, high=self.ub,
                                                    size=(self.num_particles, self.num_dimensions))
        self.particles_velocity = np.zeros((self.num_particles, self.num_dimensions))

        # 初始化粒子历史最优位置和适应度
        self.particles_best_position = np.copy(self.particles_position)

        for i in range(self.num_particles):
            fitness = objective_func(self.particles_position[i], self.objective_func_params[0], self.objective_func_params[1])
            if fitness < self.particles_best_fitness:
                self.particles_best_fitness = fitness
                self.particles_best_position[i] = np.copy(self.particles_position[i])

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(self.particles_position[i])

    def update_particles(self):
        # 更新粒子位置和速度
        w = 0.5  # 惯性权重
        c1 = 2.0  # 加速因子1
        c2 = 2.0  # 加速因子2

        for i in range(self.num_particles):
            r1 = np.random.random(self.num_dimensions)  # 随机向量1
            r2 = np.random.random(self.num_dimensions)  # 随机向量2

            # 更新速度
            self.particles_velocity[i] = w * self.particles_velocity[i] + c1 * r1 * (
                        self.particles_best_position[i] - self.particles_position[i]) + c2 * r2 * (
                                                     self.global_best_position - self.particles_position[i])

            # 限制速度范围
            self.particles_velocity[i] = np.clip(self.particles_velocity[i], self.lb, self.ub)

            # 更新位置
            self.particles_position[i] += self.particles_velocity[i]

            # 限制位置范围
            self.particles_position[i] = np.clip(self.particles_position[i], self.lb, self.ub)

    def optimize(self, stop_condition):
        """
        优化过程
        :param stop_condition: 停止条件，如果最有适应度值小于这个数则提前停止
        :return:
        """
        # 执行优化过程
        self.initialize_particles()
        best_fittness_history = []
        for _ in range(self.max_iterations):
            self.update_particles()

            for i in range(self.num_particles):
                fitness = objective_func(self.particles_position[i], self.objective_func_params[0], self.objective_func_params[1])

                if fitness < self.particles_best_fitness:
                    self.particles_best_fitness = fitness
                    self.particles_best_position[i] = np.copy(self.particles_position[i])

                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(self.particles_position[i])
            best_fittness_history.append(self.global_best_fitness)

            if self.global_best_position < stop_condition:
                break
        return self.global_best_position, best_fittness_history


# 定义QPSO类
class QPSO_improved:
    def __init__(self, params, num_particles=30, num_dimensions=10, max_iterations=100, lb=2, ub=50):
        self.num_particles = num_particles  # 粒子数量
        self.num_dimensions = num_dimensions  # 维度数量
        self.max_iterations = max_iterations  # 最大迭代次数
        self.lb = lb  # 搜索空间下界
        self.ub = ub  # 搜索空间上界
        self.objective_func_params = params  # 目标函数参数

        self.global_best_position = None  # 全局最优位置
        self.global_best_fitness = np.inf  # 全局最优适应度

        self.particles_position = None  # 粒子位置
        self.particles_velocity = None  # 粒子速度
        self.particles_best_position = None  # 粒子历史最优位置
        self.particles_best_fitness = np.inf  # 粒子历史最优适应度

        self.w = 0.5  # 初始惯性权重
        self.c1 = 2.0  # 初始加速因子1
        self.c2 = 2.0  # 初始加速因子2

    def initialize_particles(self):
        # 初始化粒子位置和速度
        self.particles_position = np.random.uniform(low=self.lb, high=self.ub,
                                                    size=(self.num_particles, self.num_dimensions))
        self.particles_velocity = np.zeros((self.num_particles, self.num_dimensions))

        # 初始化粒子历史最优位置和适应度
        self.particles_best_position = np.copy(self.particles_position)

        for i in range(self.num_particles):
            fitness = objective_func(self.particles_position[i], self.objective_func_params[0], self.objective_func_params[1])
            if fitness < self.particles_best_fitness:
                self.particles_best_fitness = fitness
                self.particles_best_position[i] = np.copy(self.particles_position[i])

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(self.particles_position[i])

    def update_particles(self, iteration):
        w_min = 0.1  # 最小惯性权重
        w_max = 0.8  # 最大惯性权重
        c_min = 0.3  # 最小加速因子
        c_max = 2.5  # 最大加速因子

        # 计算自适应的惯性权重和加速因子
        self.w = w_max - ((w_max - w_min) * iteration) / self.max_iterations
        self.c1 = c_min + ((c_max - c_min) * iteration) / self.max_iterations
        self.c2 = c_max - ((c_max - c_min) * iteration) / self.max_iterations

        for i in range(self.num_particles):
            r1 = np.random.random(self.num_dimensions)  # 随机向量1
            r2 = np.random.random(self.num_dimensions)  # 随机向量2

            # 更新速度
            self.particles_velocity[i] = self.w * self.particles_velocity[i] + self.c1 * r1 * (
                        self.particles_best_position[i] - self.particles_position[i]) + self.c2 * r2 * (
                                                     self.global_best_position - self.particles_position[i])

            # 限制速度范围
            self.particles_velocity[i] = np.clip(self.particles_velocity[i], self.lb, self.ub)

            # 更新位置
            self.particles_position[i] += self.particles_velocity[i]

            # 限制位置范围
            self.particles_position[i] = np.clip(self.particles_position[i], self.lb, self.ub)

    def optimize(self, stop_condition):
        """
        优化过程
        :param stop_condition:提前停止条件，最有适应度值小于这个数则提前停止
        :return:
        """
        # 执行优化过程
        self.initialize_particles()
        best_fitness_history = []

        for iteration in range(self.max_iterations):
            self.update_particles(iteration)

            for i in range(self.num_particles):
                fitness = objective_func(self.particles_position[i], self.objective_func_params[0], self.objective_func_params[1])

                if fitness < self.particles_best_fitness:
                    self.particles_best_fitness = fitness
                    self.particles_best_position[i] = np.copy(self.particles_position[i])

                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(self.particles_position[i])

            best_fitness_history.append(self.global_best_fitness)

            # 提前终止条件判断
            if self.global_best_position < stop_condition:
                break
        return self.global_best_position, best_fitness_history



