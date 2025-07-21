import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F

def compute_disparity(left_image, right_image):
    """计算左视图的视差图"""
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)#创建一个立体匹配对象 StereoBM，用于计算视差图。
    # 将 PyTorch 张量转换为 NumPy 数组
    left_image = left_image.cpu().numpy()  # 从 GPU 转到 CPU 并转换为 NumPy 数组
    right_image = right_image.cpu().numpy()

    # 将图像转换为 uint8 类型，并确保像素值在 [0, 255] 范围内
    left_image_np = np.clip(left_image.astype(np.float32), 0, 255).astype(np.uint8)
    right_image_np = np.clip(right_image.astype(np.float32), 0, 255).astype(np.uint8)
    # 创建一个立体匹配对象 StereoSGBM，用于计算视差图
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=0,  # 最小视差，通常设置为 0
    #     numDisparities=64,  # 视差的范围，必须是16的倍数
    #     blockSize=5,  # 匹配块的大小（通常是奇数，15是一个常用的值）
    #     P1=8 * 3 * 5 ** 2,  # 控制视差平滑的参数
    #     P2=32 * 3 * 5 ** 2,  # 控制视差平滑的参数
    #     disp12MaxDiff=1,  # 右图和左图视差之间的最大差异
    #     preFilterCap=31,  # 预滤波的最大值
    #     uniquenessRatio=50,  # 视差一致性的阈值
    #     speckleWindowSize=200,  # 斑点的窗口大小
    #     speckleRange=64  # 斑点的最大视差变化范围
    # )
    disparity = stereo.compute(left_image_np, right_image_np).astype(np.float32) / 16.0 #计算视差图，表示左图像中的像素相对于右图像的匹配偏移。
    # 将 NumPy 数组转换为 PyTorch 张量
    disparity_tensor = torch.from_numpy(disparity)

    return disparity_tensor

# def reproject_image_to_left(disparity_map, right_image):
#     """根据左视图的视差图和右视图重建左视图"""
#     # 确保 disparity_map 和 right_image 都是 Tensor 类型
#     if isinstance(right_image, torch.Tensor) is False:
#         right_image = torch.tensor(right_image)
#     if isinstance(disparity_map, torch.Tensor) is False:
#         disparity_map = torch.tensor(disparity_map)
#     print(right_image.shape)
#     print(disparity_map.shape)
#     h, w = disparity_map.shape[:2]
#
#    #print(disparity_map)
#     reprojected_left = np.zeros_like(right_image)
#
#     for y in range(h):
#         for x in range(w):
#             disparity = disparity_map[y, x]
#             x_reprojected = int(x - disparity)#通过减去视差值，将右图像中的每个像素向左移动，得到左图像中的对应位置。
#             if 0 <= x_reprojected < w:
#                 print(right_image.shape)
#                 reprojected_left[y, x] = right_image[y, x_reprojected]
#
#     return reprojected_left
def reproject_image_to_left(disparity_map, right_image):
    """
    优化后的视差重投影函数，支持以下输入形状：
    - disparity_map: (H, W) 或 (1, H, W) 或 (B, H, W)
    - right_image: (C, H, W) 或 (B, C, H, W)
    输出形状始终与right_image一致
    """
    # 统一设备
    device = disparity_map.device

    # 处理disparity_map的维度 --------------------------------------------------
    if len(disparity_map.shape) == 2:  # (H, W)
        disparity_map = disparity_map.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    elif len(disparity_map.shape) == 3:  # (B, H, W)
        disparity_map = disparity_map.unsqueeze(1)  # -> (B, 1, H, W)

    # 处理right_image的维度 ---------------------------------------------------
    if len(right_image.shape) == 3:  # (C, H, W)
        right_image = right_image.unsqueeze(0)  # -> (1, C, H, W)

    # 检查维度一致性
    B, C, H, W = right_image.shape
    assert disparity_map.shape[0] in [1, B], "Batch size mismatch"

    # 生成坐标网格 ------------------------------------------------------------
    # 创建基础网格 (B, H, W, 2)
    grid_base = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_base = torch.stack(grid_base[::-1], dim=-1).float()  # (H, W, 2) [x,y]
    grid_base = grid_base.unsqueeze(0).expand(B, H, W, 2)  # (B, H, W, 2)

    # 应用视差偏移
    grid = grid_base.clone()
    grid[..., 0] = grid_base[..., 0] - disparity_map.squeeze(1)  # x = x - disparity

    # 归一化到[-1, 1]范围
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # x方向
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # y方向

    # 执行网格采样 ------------------------------------------------------------
    reprojected = F.grid_sample(
        right_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    reprojected = reprojected.squeeze(0)

    return reprojected

def reproject_image_to_right(disparity_map, left_image):
    """
    优化后的视差重投影函数，支持以下输入形状：
    - disparity_map: (H, W) 或 (1, H, W) 或 (B, H, W)
    - left_image: (C, H, W) 或 (B, C, H, W)
    输出形状始终与right_image一致
    """
    # 统一设备
    device = disparity_map.device

    # 处理disparity_map的维度 --------------------------------------------------
    if len(disparity_map.shape) == 2:  # (H, W)
        disparity_map = disparity_map.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    elif len(disparity_map.shape) == 3:  # (B, H, W)
        disparity_map = disparity_map.unsqueeze(1)  # -> (B, 1, H, W)

    # 处理left_image的维度 ---------------------------------------------------
    if len(left_image.shape) == 3:  # (C, H, W)
        right_image = left_image.unsqueeze(0)  # -> (1, C, H, W)

    # 检查维度一致性
    B, C, H, W = left_image.shape
    assert disparity_map.shape[0] in [1, B], "Batch size mismatch"

    # 生成坐标网格 ------------------------------------------------------------
    # 创建基础网格 (B, H, W, 2)
    grid_base = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_base = torch.stack(grid_base[::-1], dim=-1).float()  # (H, W, 2) [x,y]
    grid_base = grid_base.unsqueeze(0).expand(B, H, W, 2)  # (B, H, W, 2)

    # 应用视差偏移
    grid = grid_base.clone()
    grid[..., 0] = grid_base[..., 0] - disparity_map.squeeze(1)  # x = x - disparity

    # 归一化到[-1, 1]范围
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # x方向
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # y方向

    # 执行网格采样 ------------------------------------------------------------
    reprojected = F.grid_sample(
        right_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    reprojected = reprojected.squeeze(0)

    return reprojected
# def reproject_image_to_right(disparity_map, left_image):
#     """根据左视图的视差图和右视图重建左视图"""
#     # 确保右图是 NumPy 数组，如果是 PyTorch 张量，先转为 NumPy
#     if isinstance(left_image, torch.Tensor):
#         right_image = left_image.detach().cpu().numpy()  # 如果是 tensor 类型，先移到 CPU，再转为 numpy 数组
#
#     # 确保 disparity_map 也适当格式化
#     if isinstance(disparity_map, torch.Tensor):
#         disparity_map = disparity_map.detach().cpu().numpy()  # 如果是 tensor 类型，先移到 CPU，再转为 numpy 数组
#
#     h, w = disparity_map.shape[:2]
#     reprojected_right = np.zeros_like(left_image)
#
#     for y in range(h):
#         for x in range(w):
#             disparity = disparity_map[y, x]
#             x_reprojected = int(x + disparity)
#             if 0 <= x_reprojected < w:
#                 reprojected_right[y, x_reprojected] = left_image[y, x]
#
#     return reprojected_right
# def reproject_image_to_right(disparity_map, left_image):
#     """根据左视图的视差图和右视图重建左视图"""
#     # 确保 disparity_map 和 left_image 都是 Tensor 类型
#     if isinstance(left_image, torch.Tensor) is False:
#         left_image = torch.tensor(left_image)
#     if isinstance(disparity_map, torch.Tensor) is False:
#         disparity_map = torch.tensor(disparity_map)
#
#     h, w = disparity_map.shape[:2]
#     reprojected_right = torch.zeros_like(left_image)
#     for y in range(h):
#         for x in range(w):
#             disparity = disparity_map[y, x]
#             x_reprojected = int(x + disparity.item())  # .item() 取出 tensor 的数值
#             if 0 <= x_reprojected < w:
#                 reprojected_right[y, x_reprojected] = left_image[y, x]
#
#     return reprojected_right

def mse(imageA, imageB, valid_mask):
    """计算两个图像之间的均方误差，仅考虑有效掩码中的像素"""
    # 将 numpy 数组转换为 PyTorch 张量
    imageA = torch.tensor(imageA) if not isinstance(imageA, torch.Tensor) else imageA
    imageB = torch.tensor(imageB) if not isinstance(imageB, torch.Tensor) else imageB
    #valid_mask = torch.tensor(valid_mask)
    valid_mask = valid_mask.clone().detach()

    err = torch.sum(((imageA.float() - imageB.float()) ** 2) * valid_mask) / torch.sum(valid_mask)

    return err


# class Loss_disp(nn.Module):
#     def __init__(self):
#         super(Loss_disp, self).__init__()
#
#     def forward(self, left_image_noise, right_image_noise, left_out, right_out):
#         num, width, height = left_image_noise.shape
#         error = 0
#         for i in range(num):
#             left = left_image_noise[i, :, :]
#             right = right_image_noise[i, :, :]
#             left_outsub = left_out[i, :, :]
#             right_outsub = right_out[i, :, :]
#             # 步骤1: 计算左视图的视差图
#             left = left.cpu().detach().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
#             right = right.cpu().detach().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
#             left = cv2.convertScaleAbs(left)#将图像的像素值转换为 绝对值 并进行适当的缩放
#             right = cv2.convertScaleAbs(right)
#             # left_outsub = cv2.UMat(left_outsub)
#             # right_outsub = cv2.UMat(right_outsub)
#             disparity_map_l = compute_disparity(left, right)#视差图的计算
#             disparity_map_r = compute_disparity(right, left)
#
#             # 归一化视差图以便显示
#             # disparity_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255,
#             #                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#             #
#             # # 应用颜色映射来模拟温度变化
#             # depth_map_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
#             #
#             # # 显示视差图
#             # cv2.imshow('Disparity', depth_map_colored)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#
#             # 步骤2: 使用视差图重建左视图
#             reprojected_left = reproject_image_to_left(disparity_map_l, np.squeeze(right_outsub))
#             reprojected_right = reproject_image_to_left(disparity_map_r, np.squeeze(left_outsub))
#
#             # 生成有效掩码，表示重建左视图中的非零像素
#             valid_mask_l = (reprojected_left > 0).astype(float)
#             valid_mask_r = (reprojected_right > 0).astype(float)
#
#             # 步骤3: 计算重建的左视图与原左视图之间的均方误差，仅考虑有效掩码中的像素
#             error += mse(reprojected_left, np.squeeze(left_outsub), valid_mask_l)
#             error += mse(reprojected_right, np.squeeze(right_outsub), valid_mask_r)
#
#         return error / num

class Loss_disp(nn.Module):
    def __init__(self):
        super(Loss_disp, self).__init__()
        self.mse_loss = nn.MSELoss()  # 正确初始化损失函数

    def forward(self, left_image_noise, right_image_noise, left_out, right_out):
        # print(left_image_noise.shape)
        # print(right_image_noise.shape)
        # 去掉通道维度，只保留 height 和 width
        left_image_noise = left_image_noise.squeeze(1)  # 如果通道维度是1
        right_image_noise = right_image_noise.squeeze(1)  # 如果通道维度是1

        num, width, height = left_image_noise.shape
        error = 0

        for i in range(num):
            # 获取每个图像的张量
            left = left_image_noise[i, :, :]  # shape: [width, height]
            right = right_image_noise[i, :, :]
            left_outsub = left_out[i, :, :]
            right_outsub = right_out[i, :, :]

            # 步骤1: 计算左视图的视差图
            disparity_map_l = compute_disparity(left, right)
            disparity_map_r = compute_disparity(right, left)
            # 打印 disparity_map_l 的数据类型
            #print(type(disparity_map_l))

            # 步骤2: 使用视差图重建左视图
            reprojected_left = reproject_image_to_left(disparity_map_l, right_outsub)
            reprojected_right = reproject_image_to_left(disparity_map_r, left_outsub)

            # 生成有效掩码，表示重建左视图中的非零像素
            # reprojected_left = torch.tensor(reprojected_left)
            # reprojected_right = torch.tensor(reprojected_right)
            reprojected_left = reprojected_left.clone().detach()
            reprojected_right = reprojected_right.clone().detach()
            # valid_mask_l = (reprojected_left > 0).float()
            # valid_mask_r = (reprojected_right > 0).float()

            # 步骤3: 计算重建的左视图与原左视图之间的均方误差，仅考虑有效掩码中的像素
            error += self.mse_loss(reprojected_left, left_outsub)
            error += self.mse_loss(reprojected_right, right_outsub)

        return error / num




def compute_ssim(imageA, imageB):
    """计算SSIM（结构相似性指数）"""
    # 确保 imageA 和 imageB 都是 NumPy 数组
    if isinstance(imageA, torch.Tensor):
        imageA = imageA.cpu().numpy()  # 将 imageA 从 tensor 转为 numpy
    if isinstance(imageB, torch.Tensor):
        imageB = imageB.cpu().numpy()  # 将 imageB 从 tensor 转为 numpy
        # 确保图像是浮动类型，并归一化到 [0, 1] 范围

    return ssim(imageA, imageB, data_range=imageA.max() - imageA.min())

# 主函数，进行测试并可视化
if __name__ == '__main__':
    # 读取左图和右图
    left_image_path = 'D:/Denoising10_23/clear_L/0.jpg'
    right_image_path = 'D:/Denoising10_23/clear_R/0.jpg'

    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # 计算视差图
    disparity_map_l = compute_disparity(left_image, right_image)
    disparity_map_r = compute_disparity(right_image, left_image)
    #print(type(disparity_map_l))
    # 重建图像
    reprojected_left = reproject_image_to_left(disparity_map_l, right_image)
    reprojected_right = reproject_image_to_right(disparity_map_r, left_image)
    # # test
    # left_image = torch.tensor(left_image)
    # right_image = torch.tensor(right_image)
    # reprojected_left = torch.tensor(reprojected_left)
    # reprojected_right = torch.tensor(reprojected_right)
    # left_image = left_image.unsqueeze(0)
    # right_image = right_image.unsqueeze(0)
    # reprojected_left = reprojected_left.unsqueeze(0)
    # reprojected_right = reprojected_right.unsqueeze(0)
    # disp_loss = Loss_disp()
    # loss3 = disp_loss(left_image, right_image, reprojected_left, reprojected_right)
    # print(loss3)
    # 确保 imageA 和 imageB 都是 NumPy 数组
    if isinstance(left_image, torch.Tensor):
        left_image = left_image.cpu().numpy()  # 将 imageA 从 tensor 转为 numpy
    if isinstance(reprojected_left, torch.Tensor):
        reprojected_left = reprojected_left.cpu().numpy()  # 将 imageB 从 tensor 转为 numpy
        # 确保 imageA 和 imageB 都是 NumPy 数组
    if isinstance(right_image, torch.Tensor):
        right_image = right_image.cpu().numpy()  # 将 imageA 从 tensor 转为 numpy
    if isinstance(reprojected_right, torch.Tensor):
        reprojected_right = reprojected_right.cpu().numpy()  # 将 imageB 从 tensor 转为 numpy
    # 计算SSIM
    ssim_left = ssim(left_image, reprojected_left, data_range=left_image.max() - left_image.min())
    ssim_right = ssim(right_image, reprojected_right, data_range=right_image.max() - right_image.min())

    # 输出SSIM
    print(f"SSIM between Left Image and Reprojected Left Image: {ssim_left}")
    print(f"SSIM between Right Image and Reprojected Right Image: {ssim_right}")

    # 创建 2x4 的子图布局
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 第一行: 左图、左视差图、右图、左重建图
    axes[0, 0].imshow(left_image, cmap='gray')
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(disparity_map_l, cmap='jet')
    axes[0, 1].set_title('Left Disparity Map')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(right_image, cmap='gray')
    axes[0, 2].set_title('Right Image')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(reprojected_left, cmap='gray')
    axes[0, 3].set_title('Reprojected Left Image')
    axes[0, 3].axis('off')

    # 第二行: 右图、右视差图、左图、右重建图
    axes[1, 0].imshow(right_image, cmap='gray')
    axes[1, 0].set_title('Right Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(disparity_map_r, cmap='jet')
    axes[1, 1].set_title('Right Disparity Map')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(left_image, cmap='gray')
    axes[1, 2].set_title('Left Image')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(reprojected_right, cmap='gray')
    axes[1, 3].set_title('Reprojected Right Image')
    axes[1, 3].axis('off')

    # 展示所有子图
    plt.tight_layout()
    plt.show()

    # left_images = []
    # right_images = []
    # ssim_left_values = []
    # ssim_right_values = []
    #
    # # 处理从0.jpg到100.jpg的图像
    # for i in range(101):  # 从0到100的图片
    #     left_image_path = f'D:/Denoising10_23/clear_L/{i}.jpg'
    #     right_image_path = f'D:/Denoising10_23/clear_R/{i}.jpg'
    #
    #     left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    #     right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    #
    #     left_images.append(left_image)
    #     right_images.append(right_image)
    #
    #     # 计算视差图
    #     disparity_map_l = compute_disparity(left_image, right_image)
    #     disparity_map_r = compute_disparity(right_image, left_image)
    #     # disparity_map_l, disparity_map_r = compute_disparity(
    #     #     left_image, right_image )
    #
    #     # 重建图像
    #     reprojected_left = reproject_image_to_left(disparity_map_l, right_image)
    #     reprojected_right = reproject_image_to_right(disparity_map_r, left_image)
    #
    #     # 计算SSIM
    #     ssim_left = compute_ssim(left_image, reprojected_left)
    #     ssim_right = compute_ssim(right_image, reprojected_right)
    #
    #     ssim_left_values.append(ssim_left)
    #     ssim_right_values.append(ssim_right)
    #
    #     # 输出SSIM
    #     print(f"SSIM between Left Image {i} and Reprojected Left Image: {ssim_left}")
    #     print(f"SSIM between Right Image {i} and Reprojected Right Image: {ssim_right}")
    #
    # # 输出SSIM的平均值
    # print(f"Average SSIM for Left Images: {np.mean(ssim_left_values)}")
    # print(f"Average SSIM for Right Images: {np.mean(ssim_right_values)}")
    #
    # # 可视化40-48等8个左右重建图
    # fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    #
    # # 选择40到47的图像进行展示
    # for i, idx in enumerate(range(40, 48)):
    #     left_image = left_images[idx]
    #     right_image = right_images[idx]
    #
    #     # 计算视差图
    #     disparity_map_l = compute_disparity(left_image, right_image)
    #     disparity_map_r = compute_disparity(right_image, left_image)
    #
    #     # 重建图像
    #     reprojected_left = reproject_image_to_left(disparity_map_l, right_image)
    #     reprojected_right = reproject_image_to_right(disparity_map_r, left_image)
    #
    #     # 绘制左重建图
    #     axes[i // 4, i % 4].imshow(reprojected_left, cmap='gray')
    #     axes[i // 4, i % 4].set_title(f'Reprojected Left {idx}')
    #     axes[i // 4, i % 4].axis('off')
    #
    # # 为右重建图绘制另一个子图布局
    # fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    #
    # # 绘制右重建图
    # for i, idx in enumerate(range(40, 48)):
    #     left_image = left_images[idx]
    #     right_image = right_images[idx]
    #
    #     # 计算视差图
    #     disparity_map_l = compute_disparity(left_image, right_image)
    #     disparity_map_r = compute_disparity(right_image, left_image)
    #
    #     # 重建图像
    #     reprojected_left = reproject_image_to_left(disparity_map_l, right_image)
    #     reprojected_right = reproject_image_to_right(disparity_map_r, left_image)
    #
    #     # 绘制右重建图
    #     axes2[i // 4, i % 4].imshow(reprojected_right, cmap='gray')
    #     axes2[i // 4, i % 4].set_title(f'Reprojected Right {idx}')
    #     axes2[i // 4, i % 4].axis('off')
    #
    # # 展示所有子图
    # plt.tight_layout()
    # plt.show()
