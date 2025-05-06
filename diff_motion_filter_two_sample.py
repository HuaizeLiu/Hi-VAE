import cv2
import numpy as np
from decord import VideoReader
import torchvision.transforms as transforms
import os

# input_video = "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/1037991950.mp4"
# name = "无主体，向前_2"
# # 64 窗口
# window_size = 64
# flow_mag_threshold=1.5
# direction_var_threshold=10
# # 32 窗口
# window_size = 32
# flow_mag_threshold=1.5
# direction_var_threshold=5

def visualize_flow(u, v, path):
    # 计算光流幅值和方向
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
    
    # 归一化到0-255范围
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / (2 * np.pi)  # 方向映射到色相（0-180）
    hsv[..., 1] = 255                         # 饱和度设为最大
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # 转换为BGR图像
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path, bgr)
    return bgr

def draw_flow_arrows(img, u, v, step=20, scale=1.0):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = u[y, x], v[y, x]
    
    # 绘制箭头
    vis = img.copy()
    for i in range(len(x)):
        pt1 = (x[i], y[i])
        pt2 = (int(x[i] + fx[i] * scale), int(y[i] + fy[i] * scale))
        cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
    return vis


def flow_mask(input_video, start, end, l_window_size, s_window_size, flow_mag_threshold, direction_var_threshold, direction_threshold, name=None):

    if name==None:
        name = os.path.basename(input_video).split(".")[0]
    
    # 读取相邻两帧
    video_reader = VideoReader(input_video)

    batch_index = [start]
    end_index = [end]

    frame1 = video_reader.get_batch(batch_index).asnumpy()[0]
    frame2 = video_reader.get_batch(end_index).asnumpy()[0]

    frame1 = cv2.resize(frame1, (256, 256), interpolation=cv2.INTER_LINEAR)
    frame2 = cv2.resize(frame2, (256, 256), interpolation=cv2.INTER_LINEAR)

    # for i in range(frames.shape[0]):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

    # 灰度化与高斯滤波
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5,5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5,5), 0)

    # 计算帧差并二值化
    diff = cv2.absdiff(gray1, gray2)
    _, mask_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 形态学处理优化掩膜
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_CLOSE, kernel)

    mask_colored = cv2.cvtColor(mask_diff, cv2.COLOR_GRAY2BGR)

    # 与原图叠加（红色高亮运动区域）
    mask_colored[:, :, 2] = 255  # 红色通道设为255（其他通道为0）
    overlay = cv2.addWeighted(frame1, 0.7, mask_colored, 0.3, 0)

    # 显示结果
    cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_diff_mask_1.jpg",overlay)
    cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_diff_mask_2.jpg",mask_diff)
    # cv2.imshow("Motion Overlay", overlay)


    # 计算Farneback光流
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 
        pyr_scale=0.5, levels=3, winsize=30, 
        iterations=3, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # 提取光流矢量场
    u = flow[...,0]
    v = flow[...,1]

    visualize_flow(u,v,f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_flow.jpg")
    arrow_vis = draw_flow_arrows(frame1, u, v, step=30, scale=2.0)
    cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_flow_arrow.jpg",arrow_vis)

    # 计算光流方向与幅度
    magnitude = np.sqrt(u**2 + v**2)
    direction = np.arctan2(v, u)  # 方向弧度值 [-π, π]
    
    # 新增参数
    # LARGE_WINDOW = 128  # 大窗口尺寸
    height, width = u.shape
    DIRECTION_THRESHOLD = np.pi/6  # 方向差异阈值 (30度)

    # 计算大窗口的基准运动方向
    large_window_directions = np.zeros((height//l_window_size + 1, width//l_window_size + 1))
    
    # Step 1: 计算每个大窗口的平均方向
    for y_large in range(0, height, l_window_size):
        for x_large in range(0, width, l_window_size):
            # 提取大窗口区域
            win_u_large = u[y_large:y_large+l_window_size, x_large:x_large+l_window_size]
            win_v_large = v[y_large:y_large+l_window_size, x_large:x_large+l_window_size]
            
            # 计算平均方向（矢量平均）
            avg_u = np.mean(win_u_large)
            avg_v = np.mean(win_v_large)
            avg_direction = np.arctan2(avg_v, avg_u)
            
            large_window_directions[y_large//l_window_size, x_large//l_window_size] = avg_direction

    # Step 3: 小窗口处理 255是选，0是不选
    grid_mask_camera = np.ones((height, width), dtype=np.uint8) * 255  # 初始化为全白
    grid_mask_object = np.ones((height, width), dtype=np.uint8) * 255  # 初始化为全白


    for y in range(0, height, s_window_size):
        for x in range(0, width, s_window_size):
            
            # if x == 0 and y == 32:
            #     print("yes")

            # if x == 64 and y == 192:
            #     print("yes")

            # 获取所属大窗口索引
            large_row = y // l_window_size
            large_col = x // l_window_size
            base_direction = large_window_directions[large_row, large_col]

            # 提取当前窗口的光流数据
            win_u = u[y:y+s_window_size, x:x+s_window_size]
            win_v = v[y:y+s_window_size, x:x+s_window_size]
            win_mag = magnitude[y:y+s_window_size, x:x+s_window_size]
            win_dir = direction[y:y+s_window_size, x:x+s_window_size]

            # # 判断条件1：窗口内无显著运动
            # avg_mag = np.mean(win_mag)
            # if avg_mag < flow_mag_threshold:
            #     grid_mask_camera[y:y+window_size, x:x+window_size] = 0  # 掩膜为黑色
            #     grid_mask_object[y:y+window_size, x:x+window_size] = 0  # 掩膜为黑色
            #     continue

            # 判断条件2：大小窗口方向一致性
            direction_diff = np.abs(win_dir - base_direction)
            direction_diff = np.minimum(direction_diff, 2*np.pi - direction_diff)  # 处理圆周差
            inconsistent_ratio = np.mean(direction_diff > DIRECTION_THRESHOLD)
            
            if inconsistent_ratio > direction_threshold:
                # grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 255  # 相机区域重新置黑
                grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 0  # 去除主体运动区域
            else:
                grid_mask_object[y:y+s_window_size, x:x+s_window_size] = 0  # 去除相机运动区域

            # 判断条件3：方向差异性（方差）
            dir_variance = np.var(win_dir)
            if dir_variance > direction_var_threshold:
                grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 0  # 去除快速运动的大主体
            else :
                grid_mask_object[y:y+s_window_size, x:x+s_window_size] = 0  # 去除不动的背景

            if dir_variance < 0.2:
                grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 255  # 将不动的背景保留

    # Step 4: 形态学处理优化掩膜
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    grid_mask_camera = cv2.morphologyEx(grid_mask_camera, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_grid_mask_camera.jpg",grid_mask_camera)
    grid_mask_object = cv2.morphologyEx(grid_mask_object, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_grid_mask_object.jpg",grid_mask_object)

    return grid_mask_camera, grid_mask_object
# "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/29960665.mp4" 无相机，大主体
# "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/_-9sIHCScoI.mp4" 向右移动
# "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/8652421.mp4" 无相机，小主体
# "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/27973531.mp4" 逆时针旋转
# "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/1010155946.mp4" 无主体，向右
# "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/1012177310.mp4" 无主体，向前
# "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example/1037991950.mp4" 无主体，向前
# "/mnt/pfs-gv8sxa/tts/dhg/zqy/data/webvid/test/videos_256/5665079.mp4" 无相机，大主体
# "/mnt/pfs-gv8sxa/tts/dhg/zqy/data/webvid/test/videos_256/4880366.mp4" 有相机，无主体
# "/mnt/pfs-gv8sxa/tts/dhg/zqy/data/webvid/test/videos_256/3978928.mp4" 复杂相机，小主体
# "/mnt/pfs-gv8sxa/tts/dhg/zqy/data/webvid/test/videos_256/3858893.mp4" 向左移动

if __name__ == "__main__":
    input_video = "/mnt/pfs-gv8sxa/tts/dhg/zqy/data/webvid/test/videos_256/5665079.mp4"
    name = " 无相机，大主体_2"

    # # 64 窗口
    # window_size = 64 # 窗口大小
    # flow_mag_threshold=2 # 运动幅度阈值，判断该窗口内运动强度是否足够
    # direction_var_threshold=5 # 运动方差阈值，判断该窗口内运动大小及方向是否趋近

    # 32 窗口
    s_window_size = 32
    l_window_size = 128
    flow_mag_threshold=2
    direction_var_threshold=3 # 值越小，代表对相机运动筛选越严格
    direction_threshold=0.2 # 值越小，代表对相机运动筛选越严格


    mask_1,_ = flow_mask(input_video,0,15,l_window_size, s_window_size,flow_mag_threshold,direction_var_threshold,direction_threshold,name)
    cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_grid_mask_camera_1.jpg",mask_1)
    mask_2,_ = flow_mask(input_video,15,30,l_window_size, s_window_size,flow_mag_threshold,direction_var_threshold,direction_threshold,name)
    cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_grid_mask_camera_2.jpg",mask_2)

    h,w = mask_1.shape
    mask = np.zeros_like(mask_1, dtype=np.uint8)
    for y in range(0, h, s_window_size):
        for x in range(0, w, s_window_size):
            # if np.array_equal(mask_1[y:y+s_window_size, x:x+s_window_size], mask_2[y:y+s_window_size, x:x+s_window_size]) and mask_1[y:y+s_window_size, x:x+s_window_size].any()==255:
            if np.array_equal(mask_1[y:y+s_window_size, x:x+s_window_size], mask_2[y:y+s_window_size, x:x+s_window_size]):
                if np.any(mask_1[y:y+s_window_size, x:x+s_window_size]==255):
                    mask[y:y+s_window_size, x:x+s_window_size] = 255

    cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_camera_mask.jpg",mask)

    # 1. 统计所有全0窗口的位置
    white_windows = []
    for y in range(0, h, s_window_size):
        for x in range(0, w, s_window_size):
            window = mask[y:y+s_window_size, x:x+s_window_size]
            if np.all(window == 255):  # 检查窗口是否全0
                white_windows.append((y, x))

    # 2. 计算需要保留的窗口数（32个）
    max_white_windows = 32  # 32*32*32=32768（50%）
    current_white_count = len(white_windows)

    if current_white_count > max_white_windows:
        # 3. 随机选择保留的窗口
        np.random.shuffle(white_windows)
        keep_windows = white_windows[:max_white_windows]
        
        # 4. 将未选中的全0窗口改为全255
        for y, x in white_windows[max_white_windows:]:
            mask[y:y+s_window_size, x:x+s_window_size] = 0
    
    cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_camera_mask_final.jpg",mask)

# # 生成像素坐标网格（x和y的定义）
# h, w = gray1.shape
# x, y = np.meshgrid(np.arange(w), np.arange(h))  # x坐标范围[0, w-1], y坐标范围[0, h-1]

# # 构建像素坐标+光流的点集（src_pts为原始坐标，dst_pts为光流位移后的坐标）,使用角点或光流显著性区域
# corners = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01, minDistance=10)
# src_pts = corners.reshape(-1, 2)
# dst_pts = src_pts + flow[src_pts[:,1].astype(int), src_pts[:,0].astype(int)]
# # # 剔除低运动区域
# # magnitude = np.sqrt(u + v)
# # mag_threshold = np.percentile(magnitude, 90)  # 取光流强度前10%的点
# # mask = (magnitude.ravel() > mag_threshold)
# # src_pts = src_pts[mask]
# # dst_pts = dst_pts[mask]

# # 使用RANSAC拟合全局运动模型（仿射变换）
# # M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
# M, inliers = cv2.estimateAffine2D(
#     src_pts, dst_pts, 
#     method=cv2.RANSAC,
#     ransacReprojThreshold=3.0,
#     confidence=0.99
# )

# # 计算全局运动
# print(M.shape)
# u_global = M[0,0]*x + M[0,1]*y + M[0,2]  # x坐标映射
# v_global = M[1,0]*x + M[1,1]*y + M[1,2]  # y坐标映射
# visualize_flow(u_global,v_global,f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/{name}_diff_mask_global.jpg")
# arrow_vis_global = draw_flow_arrows(frame1, u_global,v_global, step=30, scale=2.0)
# cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/{name}_diff_mask_global_arrow.jpg",arrow_vis_global)


# residual = np.sqrt((u - u_global)**2 + (v - v_global)**2)

# # 残差阈值化
# # residual_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# local = np.sqrt((u - u_global)**2 + (v - v_global)**2)
# _, mask_flow = cv2.threshold(residual_norm, 150, 255, cv2.THRESH_BINARY)

# # 优化光流掩膜
# # mask_flow = cv2.dilate(mask_flow, kernel)
# # 形态学优化
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# mask_flow = cv2.morphologyEx(mask_flow, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/{name}_diff_mask_residual_norm.jpg",residual_norm)
# cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/{name}_diff_mask_3.jpg",mask_flow)

# # 逻辑与操作融合
# final_mask = cv2.bitwise_and(mask_diff, mask_flow)

# # 区域填充（可选）
# final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

# cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/{name}_diff_mask_4.jpg",final_mask)
