import sys
from scipy.spatial.transform import Rotation as R
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import threading
import numpy as np
import cv2
import pyrealsense2 as rs
from eyeHandUI import Ui_MainWindow
from ServerSocket import Server
from CameraXYZ import CameraXYZ
import eyeHandpy
np.set_printoptions(suppress=True)  # 禁用科学计数法
class MyWindows(QMainWindow, Ui_MainWindow,Server,CameraXYZ):
    def __init__(self):
        super(MyWindows, self).__init__()
        self.xyzwprList ,self.autoProgressBarValue,self.displayCalcResultList,self.displayCheckResultList=[],0,'',''
        self.timerUpdata = QTimer(self)
        self.timerUpdata.start(50)
        self.timerUpdata.timeout.connect(self.updateDataDF)
        self.setupUi(self)
        self.connectBtn.clicked.connect(self.openIPconnect)#打开tcp通讯
        self.disconnectBtn.clicked.connect(self.closeIPconnect)#关闭tcp通讯
        self.openCameraBtn.clicked.connect(self.openCameraDF)
        self.closeCameraBtn.clicked.connect(self.closeCameraDF)
        self.autoBtn.clicked.connect(self.autoRunDF)
        self.readposeFileDF()
        #初始化按钮状态
        self.autoBtn.setEnabled(False)
        self.closeCameraBtn.setEnabled(False)
        self.disconnectBtn.setEnabled(False)
    def readposeFileDF(self):
        with open('pose/pose.txt','r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                self.cleaned_line = line.strip()  # 去除行尾换行符
                self.xyzwprList.append(self.cleaned_line)
                # 将字符串中的数据转换为浮点数列表
                self.cleaned_data = [list(map(float, filter(None, item.split(',')))) for item in self.xyzwprList]
        self.xyzwprList = self.cleaned_data
    def openIPconnect(self):
        self.start_server(self.ipArr.displayText(),int(self.ipPort.displayText()))
        self.connectBtn.setEnabled(False)
        self.disconnectBtn.setEnabled(True)
    def closeIPconnect(self):
        self.closeclient()
        self.connectBtn.setEnabled(True)
        self.disconnectBtn.setEnabled(False)
    def updateDataDF(self):
        # 仅在结果内容变化时更新显示
        if self.client_sock_flag:
            if self.connectLabel.text() !='已连接':
                self.connectLabel.setText('已连接')
                #打开自动按钮
                self.autoBtn.setEnabled(True)
        else:
            if self.connectLabel.text() != '未连接':
                self.connectLabel.setText('未连接')
                self.autoBtn.setEnabled(False)
        #进度条更新
        if self.autoProgressBar.text() != self.autoProgressBarValue:
            self.autoProgressBar.setValue(self.autoProgressBarValue)
        #手眼标定结果
        if self.displayCalcResult.toPlainText() != self.displayCalcResultList:
            self.displayCalcResult.setText(self.displayCalcResultList)
        #手眼标定验算结果
        if self.displayCheckResult.toPlainText() != self.displayCheckResultList:
            self.displayCheckResult.setText(self.displayCheckResultList)
    def update_frame(self):
        #从相机类获取对齐的深度彩色帧
        rgb_qimage,depth_qimage=self.get_camera_frame()
        # 将图像转换为Qt可接受的格式
        depth_qimage = self.convert_cv2_to_qimage(depth_qimage)
        rgb_qimage = self.convert_cv2_to_qimage(rgb_qimage)
        # 显示深度图像和RGB图像
        self.depth_label.setPixmap(QPixmap.fromImage(depth_qimage))
        self.rgb_label.setPixmap(QPixmap.fromImage(rgb_qimage))
    def openCameraDF(self):
        self.closeCameraBtn.setEnabled(True)
        self.openCameraBtn.setEnabled(False)
        self.get_camera_pipe()
        # 定时器更新帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    def closeCameraDF(self):
        self.pipeline.stop()
        self.timer.stop()
        # 关闭深度图像和RGB图像
        self.depth_label.setText('未连接深度相机')
        self.rgb_label.setText('未连接彩色相机')
        self.closeCameraBtn.setEnabled(False)
        self.openCameraBtn.setEnabled(True)
    def convert_cv2_to_qimage(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    def xyzwprCount(self,i):
        return ','.join(map(str, self.xyzwprList[i]))
    def sendPlane1(self):
        self.send_data(self.xyzwprCount(0))
        self.moveReach()
    #单次移动
    def movePathThread(self):
        while True:
            if self.dataRec:
                self.dataRec = ''
                break
    #移动到达指定位置
    def moveReach(self):
        self.dataRec = ''
        movethread = threading.Thread(target=self.movePathThread)
        movethread.start()
    def autoRunDF(self):
        self.autoBtn.setEnabled(False)
        autoRunThread = threading.Thread(target=self.autoRunThreadDF)
        autoRunThread.start()
    def autoRunThreadDF(self):
        self.eyeHandcomBox.setEnabled(False)
        for i in range(0,len(self.xyzwprList)):
            self.autoProgressBarValue=int((i+1)/len(self.xyzwprList)*100)
            self.send_data(self.xyzwprCount(i))
            #等待到达
            self.movePathThread()
            #拍照
            self.shotColorFrame(i)
        print(self.eyeHandcomBox.currentText())
        if self.eyeHandcomBox.currentText() == '眼在手上':
            self.displayCalcResultList,self.displayCheckResultList = eyeHandpy.eyeHandCalc((11, 8), 10, self.eyeHandcomBox.currentText())
        elif self.eyeHandcomBox.currentText() == '眼在手外':
            self.displayCalcResultList,self.displayCheckResultList = eyeHandpy.eyeHandCalc((11, 8), 10, self.eyeHandcomBox.currentText())
        self.eyeHandcomBox.setEnabled(True)
if  __name__ == '__main__':
    app = QApplication(sys.argv)
    my_windows = MyWindows()  # 实例化对象
    my_windows.setFixedSize(1850, 1020)
    my_windows.setWindowTitle("视觉机器人手眼标定精度分析软件")
    my_windows.show()  # 显示窗口
    sys.exit(app.exec_())
