import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
np.set_printoptions(suppress=True)  # suppress参数用于禁用科学计数法

# 用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = np.dot(np.dot(Rz, Ry), Rx)
    return R
def rotationMatrixToEulerAngles(R):
    """
    将旋转矩阵转换为欧拉角 (rx, ry, rz)，以 ZYX 顺序为标准。
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6  # 检查是否接近奇异情况

    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # 处理奇异情况
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0

    return np.array([rx, ry, rz])
# 用于根据位姿计算变换矩阵
def pose_robot( x, y,z , Tx, Ty, Tz):  # 注意输入顺序！！！！！！
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    # print(R)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    return RT1
# 用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(rot_vector, trans):
    ####
    rotMat = cv2.Rodrigues(rot_vector)[0]
    t = np.array(trans).reshape(3, 1)  # 确保 t 是一个二维数组
    RT = np.column_stack((rotMat, t))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    return RT

robot_pose = []
# 指定字符集为 utf-8-sig 是为了自动处理BOM字符，并消除\ufeff的出现
with open(r'.\pose\pose.txt', 'r', encoding='utf-8-sig') as f:
    # 使用索引读取第3-5行
    lines = f.readlines()
    # 删除换行符，遍历读取
    for line in lines:
        cleaned_line = line.strip()  # 去除行尾换行符
        robot_pose.append(cleaned_line)
        # 将字符串中的数据转换为浮点数列表
        templist = [list(map(float, filter(None, item.split(',')))) for item in robot_pose]
    robot_pose = templist


def eyeHandCalc(chessboard_size,square_size,eyeHandmethod):
    # 定义棋盘格的大小和角点的数量
    # chessboard_size = (11, 8)  # 角点数量 (行数-1, 列数-1)
    # square_size = 10  # 棋盘格每个方格的大小，单位为mm

    # 准备对象点坐标，例如 (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 用于存储所有图像的对象点和图像点
    objpoints = []  # 3d 点在现实世界中的位置
    imgpoints = []  # 2d 点在图像平面中的位置

    # 读取棋盘格图片并检测角点
    for i in range(0, len(robot_pose)):  # 假设有18张图片，文件名为1.jpg, 2.jpg, ..., 18.jpg
        # img_path = f'calcImg/Calibration_{i}.png '#Calibration_
        img_path = f'calcImg/{i}.jpg'  # 仿真
        img = cv2.imread(img_path)
        if img is None:
            print(f"图像文件 {img_path} 不存在或无法读取")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # 提升角点精度到亚像素级别
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            # print(i)
            # 画出棋盘格角点
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            # cv2.imshow('Chessboard', img)
            # cv2.waitKey(300)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                       None, None)
    good_picture = [i for i in range(len(robot_pose))]
    # 计算标定板到相机的变换矩阵
    R_all_chess_to_cam_1 = []
    T_all_chess_to_cam_1 = []
    for i in good_picture:
        chess2cam_RT = get_RT_from_chessboard(rvecs[i], tvecs[i])
        R_all_chess_to_cam_1.append(chess2cam_RT[:3, :3])  # 旋转矩阵
        T_all_chess_to_cam_1.append(chess2cam_RT[:3, 3].reshape((3, 1)))  # 平移向量

    # 计算法兰末端位姿与相机的变换矩阵
    R_all_end_to_base_1 = []
    T_all_end_to_base_1 = []
    for i in good_picture:
        end2robot_RT = pose_robot(robot_pose[i][3], robot_pose[i][4], robot_pose[i][5],
                                  robot_pose[i][0], robot_pose[i][1], robot_pose[i][2])
        # 眼在手上
        if eyeHandmethod == '眼在手上':
            R_all_end_to_base_1.append(end2robot_RT[:3, :3])
            T_all_end_to_base_1.append(end2robot_RT[:3, 3].reshape((3, 1)))
        # 眼在手外
        if eyeHandmethod == '眼在手外':
            rb2e = np.linalg.inv(end2robot_RT)
            R_all_end_to_base_1.append(rb2e[:3, :3])
            T_all_end_to_base_1.append(rb2e[:3, 3].reshape((3, 1)))
    # 手眼标定
    R, T = cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, R_all_chess_to_cam_1, T_all_chess_to_cam_1,
                                method=cv2.CALIB_HAND_EYE_TSAI)
    RT = np.column_stack((R, T))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))  # 即相机到机械臂末端法兰变换矩阵
    print('相机相对于末端的变换矩阵为：')
    print(RT)
    chess2base_T = []
    chess2base_theta = []
    # 固定的棋盘格相对于机器人基坐标系位姿不变，对结果验证，原则上来说，每次结果相差较小
    for i in range(len(good_picture)):
        RT_end_to_base = np.column_stack((R_all_end_to_base_1[i], T_all_end_to_base_1[i]))
        RT_end_to_base = np.row_stack((RT_end_to_base, np.array([0, 0, 0, 1])))
        RT_chess_to_cam = np.column_stack((R_all_chess_to_cam_1[i], T_all_chess_to_cam_1[i]))
        RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
        RT_cam_to_end = np.column_stack((R, T))
        RT_cam_to_end = np.row_stack((RT_cam_to_end, np.array([0, 0, 0, 1])))
        RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam  # 即为固定的棋盘格相对于机器人基坐标系位姿不变
        RT_chess_to_base = np.linalg.inv(RT_chess_to_base)
        chess2base_T.append(RT_chess_to_base[:, -1])
        chess2base_theta.append(RT_chess_to_base[:3, :3])
    chess2base_T_List = []
    # 棋盘格相对于机械臂基座标XYZ不变，对结果验证
    for i in range(len(chess2base_T)):
        # print(rotationMatrixToEulerAngles(chess2base_theta[i]) / np.pi * 180)
        print(f"{i}:", chess2base_T[i])
        chess2base_T_List.append(chess2base_T[i].tolist())
    print('chess2base_T_List',chess2base_T_List)
    return '\n'.join([' '.join(f'{x:.6f}' for x in row)for row in RT]),'\n'.join([' '.join(f'{x:.6f}' for x in row)for row in chess2base_T_List])

# print(np.array(chess2base_T[1:3]))
# # 提取 y, x, z 数据
# y_data = [point[0] for point in chess2base_T]
# x_data = [point[1] for point in chess2base_T]
# z_data = [point[2] for point in chess2base_T]
#
# # 创建一个新的3D绘图对象
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制3D散点图
# ax.scatter(x_data, y_data, z_data, c='r', marker='o')
#
# # 设置坐标轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# # 显示图形
# plt.show()