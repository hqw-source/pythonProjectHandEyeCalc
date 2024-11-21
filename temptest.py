import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 示例数据
chess2base_T = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]

class Plot3DCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # 创建 matplotlib 图形对象
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        super().__init__(self.fig)  # 初始化父类

    def plot(self, x_data, y_data, z_data):
        # 清除当前绘图
        self.ax.clear()

        # 绘制 3D 散点图
        self.ax.scatter(x_data, y_data, z_data, c='r', marker='o')

        # 设置坐标轴标签
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

        # 刷新绘图
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建布局
        self.layout = QVBoxLayout(self.central_widget)

        # 创建绘图画布
        self.canvas = Plot3DCanvas(self)
        self.layout.addWidget(self.canvas)

        # 提取 y, x, z 数据
        y_data = [point[0] for point in chess2base_T]
        x_data = [point[1] for point in chess2base_T]
        z_data = [point[2] for point in chess2base_T]

        # 绘制图形
        self.canvas.plot(x_data, y_data, z_data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
