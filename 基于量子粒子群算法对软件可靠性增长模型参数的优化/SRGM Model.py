import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False


def srgm_log_linear_model(t, a=0.1, b=0.2):
    """
    SRGM的对数线性模型函数
    :param t: 时间序列
    :param a: 模型参数a
    :param b: 模型参数b
    :return: 可靠度序列
    """
    return 1 - (a + b * np.log(t))


def weibull_model(t, a=0.1, b=2, c=0.5):
    """
    韦伯模型函数
    :param t: 时间序列
    :param shape: 模型参数shape（形状参数）
    :param scale: 模型参数scale（尺度参数）
    :return: 可靠度序列
    """
    return 1- a * (1 - math.e**(-(t / b) ** c))


def lognormal_model(t, a=0.3, b=0.5):
    """
    对数正态模型函数
    :param t: 时间序列
    :param mu: 对数正态分布的均值参数
    :param sigma: 对数正态分布的标准差参数
    :return: 可靠度序列
    """
    return a/(1+b*np.log(t))


# 生成时间序列
t = np.linspace(0.001, 100.001, 100)  # 从0到100均匀分布的100个时间点



# 计算可靠度序列
reliability = lognormal_model(t)

# 绘制图像
plt.plot(t, reliability)
plt.xlim(1, 100)
plt.xlabel('时间')
plt.ylabel('故障率')
plt.grid(True)
#plt.savefig('../图片/Lognormal Model.png', dpi=300)
plt.show()

