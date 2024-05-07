import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_excel('./Static/best.xlsx')
# 提取列数据
loss = df['loss']
val_loss = df['val_loss']
acc = df['acc']
val_acc = df['val_acc']

# 绘制第一张图：loss和val_loss随迭代次数的变化
plt.figure(figsize=(10, 6))
plt.plot(loss, label='测试集')
plt.plot(val_loss, label='验证集')
plt.xlabel('迭代次数')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# plt.savefig('../图片/Loss随迭代次数变化.png', dpi=300)
plt.show()


# 绘制第二张图：acc和val_acc随迭代次数的变化
plt.figure(figsize=(10, 6))
plt.plot(acc, label='测试集')
plt.plot(val_acc, label='验证集')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)
# plt.savefig('../图片/准确率随迭代次数变化.png', dpi=300)
plt.show()
