import pyrealsense2 as rs
import numpy as np
import cv2
import threading
class CameraXYZ:
    def get_camera_pipe(self):
        # 初始化深度相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # 开启深度相机
        self.pipeline.start(self.config)
        # 创建伪彩色映射
        self.color_map = cv2.COLORMAP_JET
        # 创建对齐对象，将深度帧与颜色帧对齐
        self.align = rs.align(rs.stream.color)
    def get_camera_frame(self):
        # 等待新的帧
        self.frames = self.pipeline.wait_for_frames()
        # 对齐深度帧和颜色帧
        self.aligned_frames = self.align.process(self.frames)
        self.depth_frame = self.aligned_frames.get_depth_frame()
        self.color_frame = self.aligned_frames.get_color_frame()
        if not self.depth_frame or not self.color_frame:
            return
        # 创建深度图像的伪彩色映射器
        colorizer = rs.colorizer()
        # 应用伪彩色映射
        depth_colormap = np.asanyarray(colorizer.colorize(self.depth_frame).get_data())
        # 获取RGB图像并转换为numpy数组
        color_image = np.asanyarray(self.color_frame.get_data())

        return color_image,depth_colormap
    def get_camera_coordinates(self, pixel_x, pixel_y):
        """
        获取相机坐标系下给定像素位置的三维坐标。
        :param pixel_x: 像素的X坐标
        :param pixel_y: 像素的Y坐标
        :return: 三维坐标 (x, y, z)
        """
        if not self.depth_frame:
            print("无法获取深度帧")
            return None
        # 获取深度图像的深度值
        depth_value = self.depth_frame.get_distance(pixel_x, pixel_y)
        if depth_value == 0:
            print("该点无效或超出深度范围")
            return None
        # 获取相机的内参
        depth_intrinsics = self.depth_frame.profile.as_video_stream_profile().intrinsics
        # 将像素坐标转换为相机坐标
        camera_coordinates = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel_x, pixel_y], depth_value * 1000)
        print(f"{camera_coordinates}")
        return camera_coordinates
    def get_edge_point_threadDF(self):
        coord_mm_MAX_frames = []  # 用于存储每帧的最大值
        for frame_idx in range(10):  # 采集十帧
            coord_mm_MAX = [0, 0, 0]  # 每帧临时最大值
            pxMAX, pyMAX = 0, 0  # 初始化最大值位置
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 表示左右下、上的方向
            offset = 100  # 偏移量
            for dx, dy in directions:
                for i in range(1, offset + 1):
                    px = 640 + i * dx
                    py = 360 + i * dy
                    if 0 <= px < 1280 and 0 <= py < 720:  # 确保在图像范围内
                        camera_coordinate = self.get_camera_coordinates(px, py)

                        if camera_coordinate:
                            coord_mm = [round(c, 3) for c in camera_coordinate]  # 转换为毫米
                            print()
                            if coord_mm[2] > coord_mm_MAX[2] and coord_mm[2] < 500:
                                coord_mm_MAX = coord_mm
                                pxMAX, pyMAX = px, py  # 记录最大值位置
            coord_mm_MAX_frames.append(coord_mm_MAX)  # 存储每帧的最大值
        print('coord_mm_MAX_frames', coord_mm_MAX)
        # 计算十帧数据的平均最大值
        if coord_mm_MAX_frames:
            avg_coord = np.mean(coord_mm_MAX_frames, axis=0)
            print(f"[{avg_coord[0]:.3f}, {avg_coord[1]:.3f}, {avg_coord[2]:.3f}],")
            return avg_coord.tolist()
    def get_edge_pointDF(self):
        Tempthread = threading.Thread(target=self.get_edge_point_threadDF)
        Tempthread.start()
        pass
    def shotColorFrame(self,i):
        color_frame,_ = self.get_camera_frame()
        print('拍照')
        cv2.imwrite(f'calcImg/Calibration_{i}.png',color_frame)

if __name__ == '__main__':
    # 创建 CameraXYZ 对象
    camera = CameraXYZ()
    camera.get_camera_pipe()  # 初始化相机
    # 打开窗口显示相机图像
    cv2.namedWindow("RGB Image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth Image", cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            # 获取RGB和深度图像帧
            images = camera.get_camera_frame()
            if images:
                depth_qimage, rgb_qimage = images

                # 将Qt格式的图像转换为OpenCV格式（BGR），并显示
                rgb_image = cv2.cvtColor(np.array(rgb_qimage), cv2.COLOR_RGB2BGR)
                depth_image = cv2.cvtColor(np.array(depth_qimage), cv2.COLOR_RGB2BGR)

            # 按键处理
            key = cv2.waitKey(1)
            if key == 27:  # ESC 退出
                break
            elif key == ord('e'):  # 按 'e' 键获取边缘点
                print("正在计算边缘点...")
                result, pxMAX, pyMAX = camera.get_edge_point_threadDF()
                if result:
                    # 确保 pxMAX 和 pyMAX 是整数
                    pxMAX, pyMAX = int(pxMAX), int(pyMAX)
                    # 在RGB图像上绘制绿色圆圈标记边缘点
                    cv2.circle(rgb_image, (pxMAX, pyMAX), 5, (0, 255, 0), -1)
                    print(f"边缘点位置: ({pxMAX}, {pyMAX})")
            # 显示更新后的图像
            cv2.imshow("RGB Image", rgb_image)
            cv2.imshow("Depth Image", depth_image)

    finally:
        cv2.destroyAllWindows()
        camera.pipeline.stop()
