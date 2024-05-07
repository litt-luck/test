import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 测试数据（累积错误数量）
test_data = np.array([2, 8, 14, 22, 30, 35, 42, 48, 52, 60, 65, 70, 78, 82, 89, 95, 100, 105, 110, 118])


def read_arrf(file):
    """
    读取arff格式文件
    :param file: 文件的绝对路径
    :return:
    """
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    return df


def goel_okumoto(t, a, b):
    """
    Goel-Okumoto模型公式，TEF参数表示为(a,b)
    :param t:时间
    :param a:模型参数
    :param b:模型参数
    :return:
    """
    return a * (1 - np.exp(-b * t))


def fitness_function(params):
    """
    适应度函数
    :param params:TEF参数，(a,b)
    :return: 均方误差
    """
    # 适应度函数
    a, b = params
    predictions = goel_okumoto(np.arange(1, len(test_data) + 1), a, b)
    mse = np.mean((predictions - test_data) ** 2)
    return -mse


def initialize_particles(n_particles, n_dimensions, search_space):
    """
    初始化粒子群
    :param n_particles: 种群大小
    :param n_dimensions: 维度
    :param search_space: 搜索空间
    :return:
    particles：种群
    """
    particles = np.random.uniform(search_space[0], search_space[1], (n_particles, n_dimensions))
    return particles


# 更新粒子位置
def update_positions(particles, personal_best_positions, global_best_position, alpha):
    """
    更新粒子位置
    :param particles:
    :param personal_best_positions: 个体的最优位置
    :param global_best_position: 全局最优位置
    :param alpha: 学习因子
    :return:
    """
    n_particles, n_dimensions = particles.shape
    # 计算每个粒子与全局最优解之间的距离
    distances = global_best_position - particles
    # 更新粒子位置
    new_positions = particles + alpha * distances * np.random.rand(n_particles, n_dimensions)
    # 限制粒子位置在搜索空间内
    new_positions = np.clip(new_positions, search_space[0], search_space[1])
    return new_positions


def qpso_improved(n_particles, n_dimensions, search_space, n_iterations, alpha):
    """
    改进的QPSO算法优化过程
    :param n_particles:粒子群大小
    :param n_dimensions:维度
    :param search_space:搜索空间
    :param n_iterations:迭代次数
    :param alpha:学习因子
    :return:
    global_best_position:最优TEF参数
    global_best_fitness_value:最优适应度值，负MSE
    global_best_fitness_values:全局最优适应度值列表，用于绘图
    """
    # 初始化粒子群
    particles = initialize_particles(n_particles, n_dimensions, search_space)
    personal_best_positions = np.copy(particles)
    personal_best_fitness_values = np.array([fitness_function(p) for p in personal_best_positions])
    # 初始化全局最优解
    global_best_position = personal_best_positions[np.argmax(personal_best_fitness_values)]
    global_best_fitness_value = np.max(personal_best_fitness_values)

    # 用于绘图的全局最优适应度值列表
    global_best_fitness_values = [global_best_fitness_value.item()]

    print(f'**********改进QPSO算法*************')

    # 迭代优化
    for i in range(n_iterations):
        # 更新粒子位置
        particles = update_positions(particles, personal_best_positions, global_best_position, alpha)

        # 计算新位置的适应度值
        fitness_values = np.array([fitness_function(p) for p in particles])

        # 更新个体最优解
        better_personal_mask = fitness_values > personal_best_fitness_values
        personal_best_positions[better_personal_mask] = particles[better_personal_mask]
        personal_best_fitness_values[better_personal_mask] = fitness_values[better_personal_mask]

        # 更新全局最优解
        current_best_index = np.argmax(fitness_values)
        if fitness_values[current_best_index] > global_best_fitness_value:
            global_best_position = particles[current_best_index]
            global_best_fitness_value = fitness_values[current_best_index]
        global_best_fitness_values.append(global_best_fitness_value.item())

        # print(f'迭代次数: {i + 1}, 全局最优适应度值: {global_best_fitness_value}')

    return global_best_position, global_best_fitness_value, global_best_fitness_values


def qpso_traditional(n_particles, n_dimensions, search_space, n_iterations, alpha):
    """
    传统QPSO算法优化过程
    :param n_particles:粒子群大小
    :param n_dimensions:维度
    :param search_space:搜索空间
    :param n_iterations:迭代次数
    :param alpha:学习因子
    :return:
    global_best_position:最优TEF参数
    global_best_fitness_value:最优适应度值，负MSE
    global_best_fitness_values:全局最优适应度值列表，用于绘图
    """
    particles = initialize_particles(n_particles, n_dimensions, search_space)
    personal_best_positions = np.copy(particles)
    personal_best_fitness_values = np.array([fitness_function(p) for p in personal_best_positions])
    global_best_position = personal_best_positions[np.argmax(personal_best_fitness_values)]
    global_best_fitness_value = np.max(personal_best_fitness_values)

    global_best_fitness_values = [global_best_fitness_value.item()]
    print(f'**********传统QPSO算法*************')
    for i in range(n_iterations):
        # 在传统QPSO中，使用固定的学习因子0.5
        particles = update_positions(particles, personal_best_positions, global_best_position, 0.5)
        fitness_values = np.array([fitness_function(p) for p in particles])

        better_personal_mask = fitness_values > personal_best_fitness_values
        personal_best_positions[better_personal_mask] = particles[better_personal_mask]
        personal_best_fitness_values[better_personal_mask] = fitness_values[better_personal_mask]

        current_best_index = np.argmax(fitness_values)
        if fitness_values[current_best_index] > global_best_fitness_value:
            global_best_position = particles[current_best_index]
            global_best_fitness_value = fitness_values[current_best_index]

        global_best_fitness_values.append(global_best_fitness_value.item())

        # print(f'迭代次数: {i + 1}, 全局最优适应度值: {global_best_fitness_value}')

    return global_best_position, global_best_fitness_value, global_best_fitness_values


if __name__ == "__main__":
    # 设置QPSO参数
    n_particles = 30
    n_dimensions = 2
    search_space = np.array([[0, 0], [150, 1]])
    n_iterations = 100
    alpha = 0.5
    # 执行改进的QPSO优化
    best_position, best_fitness_value, global_best_fitness_values = qpso_improved(n_particles, n_dimensions,
                                                                                  search_space, n_iterations, alpha)
    a_opt, b_opt = best_position
    print(f'最优TEF参数（a, b）: ({a_opt}, {b_opt}), 最优适应度值（负MSE）: {best_fitness_value}')

    # # 绘制优化过程
    # plt.plot(range(n_iterations + 1), global_best_fitness_values)
    # plt.xlabel('Iteration')
    # plt.ylabel('Best Fitness Value (Negative MSE)')
    # plt.title('QPSO Optimization of TEF Parameters in Goel-Okumoto Model')
    # plt.show()

    # 使用传统QPSO算法优化TEF参数
    best_position_traditional, best_fitness_value_traditional, global_best_fitness_values_traditional = qpso_traditional(
        n_particles, n_dimensions, search_space, n_iterations, alpha
    )
    a_opt_traditional, b_opt_traditional = best_position_traditional
    print(
        f'最优TEF参数（a, b）: ({a_opt_traditional}, {b_opt_traditional}), 最优适应度值（负MSE）: {best_fitness_value_traditional}')

    # 绘制改进QPSO和传统QPSO的性能比较图
    plt.plot(range(n_iterations + 1), global_best_fitness_values, label='改进QPSO')
    plt.plot(range(n_iterations + 1), global_best_fitness_values_traditional, label='传统QPSO')
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('最佳适应度值 (负均方误差)')
    # plt.title('改进QPSO与传统QPSO优化Goel-Okumoto模型中的TEF参数性能比较')
    plt.legend()
    # plt.savefig("改进QPSO与传统QPSO优化Goel-Okumoto模型中的TEF参数性能比较.png", dpi=300)
    plt.show()
