import os
from skimage import io, filters, color
import numpy as np
import matplotlib.pyplot as plt
import cv2


def bar_show(variances):
    data = np.array(variances)
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) # 归一化数据到[0,1]

    # 创建一个248x248的黑色图像（所有像素值初始设为0，即黑色）
    img = np.zeros((248, 248), dtype=np.uint8)

    # 计算每个分割的宽度
    segment_width = img.shape[1] // 31

    # 将数据映射到灰度级（0-255）
    data_scaled = (data * 255)

    # 填充每个段的颜色
    for i, value in enumerate(data_scaled):
        img[:, i * segment_width:(i + 1) * segment_width] = value

    # 使用OpenCV显示图像
    # cv2.imshow('Grayscale Representation', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def laplace_blur():
    image_path = 'D:\\holopix_248\\train\\left view_noise(75)\\1.jpg'
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image) # 转换为灰度图像
    laplace_image = filters.laplace(gray_image) # 应用拉普拉斯算子来检测图像的边缘，输出图像中每个点的变化情况

    variance = np.var(laplace_image) #计算拉普拉斯图像的方差，反映图像的变化程度或模糊程度。

    # plt.imshow(laplace_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    print(variance)


def adaptive_threshold_gradient(image_path):
    # 读取图像并转换为灰度图像
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 计算图像的Sobel梯度
    gradient = filters.sobel(image)

    # 计算全局梯度阈值（例如，使用全局梯度的中位数）
    global_threshold = np.median(gradient)

    # 创建一个空数组，用于存储自适应阈值处理后的图像
    adaptive_threshold_image = np.zeros_like(gradient)

    # 定义局部区域的大小
    block_size = 100

    # 遍历图像，对每个局部区域进行处理
    for y in range(0, gradient.shape[0], block_size):
        for x in range(0, gradient.shape[1], block_size):
            # 提取局部区域
            block = gradient[y:y + block_size, x:x + block_size]
            # 计算局部梯度阈值（例如，使用局部梯度的平均值）
            local_threshold = np.mean(block)
            # 根据局部梯度阈值判断模糊区域
            adaptive_threshold_image[y:y + block_size, x:x + block_size] = block > local_threshold

    # 将处理结果转换为二值图像
    binary_image = (adaptive_threshold_image > 0).astype(np.uint8) * 255

    # 显示结果
    # cv2.imshow('Adaptive Threshold Gradient', gradient)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return gradient


def right_calculate_weights(idx, image, n, threshold, min_weight=0.03, max_weight=1.0):
    # 读取图像并转换为灰度图像
    height, width = image.shape
    region_width = width // n

    variances = []
    # 计算每个区域的方差
    for i in range(n):
        start_col = i * region_width
        end_col = start_col + region_width
        region = image[:, start_col:end_col]
        laplace_region = filters.laplace(region)
        variance = np.var(laplace_region)
        variances.append(variance)

    bar_show(variances)
    # 检查所有区域的方差是否都小于阈值
    all_below_threshold = all(var < threshold for var in variances)

    # 初始化权重
    weights = np.ones(width) * min_weight

    # 从右向左查找连续方差小于阈值的区域
    consecutive_below_threshold = 0
    for variance in reversed(variances):
        if variance < threshold:
            consecutive_below_threshold += 1
        else:
            break

    # 确定锚点位置
    if consecutive_below_threshold > 0:
        anchor = n - consecutive_below_threshold
        # 计算权重
        for i in range(anchor * region_width):
            weights[i] = min_weight
        for i in range(anchor * region_width, width):
            weights[i] = ((i - anchor * region_width) / (width - anchor * region_width)) * (
                        max_weight - min_weight) + min_weight
    else:
        # 如果整个图像的方差都小于阈值，则不设置锚点
        anchor = n

    # 将权重映射到0到255的范围
    weights_mapped = np.interp(weights, (min_weight, max_weight), (min_weight*255, 255))
    weight_image = np.tile(weights_mapped, (height, 1))

    # 绘制权重图
    # plt.imshow(weight_image, cmap='gray')
    # plt.title('Weight Image')
    # plt.axis('off')
    # plt.show()
    save_path = os.path.join('right view_weight', f'{idx}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保文件夹存在
    cv2.imwrite(save_path, weight_image)

    # 保存权重图像为JPEG文件
    # cv2.imwrite(os.path.join('right view_weight', idx), weight_image)


def left_calculate_weights(idx, image, n, threshold, min_weight=0.03, max_weight=1.0):
    height, width = image.shape
    region_width = width // n

    variances = []
    for i in range(n):
        start_col = i * region_width
        end_col = start_col + region_width
        region = image[:, start_col:end_col]
        laplace_region = filters.laplace(region)
        variance = np.var(laplace_region)
        variances.append(variance)

    bar_show(variances)
    # 初始化权重
    weights = np.ones(width) * max_weight

    # 从左向右查找连续方差小于阈值的区域
    last_below_threshold = -1  # 初始化为-1，表示没有找到符合条件的区域
    for i, variance in enumerate(variances):
        if variance < threshold:
            last_below_threshold = i  # 更新最后一个小于阈值的区域索引
        else:
            break

    # 确定锚点位置
    if last_below_threshold != -1:
        anchor = last_below_threshold + 1  # 锚点是连续小于阈值区域的最后一个区域的右侧
        # 从锚点位置到图像右侧的权重设置为0.03
        for i in range(anchor * region_width, width):
            weights[i] = min_weight
        # 从图像左侧到锚点位置的权重从1降低到0.03
        for i in range(0, anchor * region_width):
            weights[i] = (i / (anchor * region_width)) * (min_weight - max_weight) + max_weight

    # 将权重映射到0到255的范围，并转换为uint8类型
    weights_mapped = np.interp(weights, (min_weight, max_weight), (255 * min_weight, 255)).astype(np.uint8)
    weight_image = np.tile(weights_mapped, (height, 1))

    # 绘制权重图
    # plt.imshow(weight_image, cmap='gray')
    # plt.title('Weight Image Based on Variance from Left to Right')
    # plt.axis('off')
    # plt.show()

    save_path = os.path.join('left view_weight', f'{idx}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保文件夹存在
    cv2.imwrite(save_path, weight_image)
    # 保存权重图像为JPEG文件
    # cv2.imwrite(os.path.join('left view_weight', idx), weight_image)


# def evaluate_blur_by_regions(image, n):
#     # 获取图像的高度和宽度
#     height, width = image.shape
#     # 计算每个区域的宽度
#     region_width = width // n
#     weight = np.ones(width)
#
#     # 遍历每个区域
#     for i in range(n):
#         # 计算当前区域的起始和结束列索引
#         start_col = i * region_width
#         end_col = (i + 1) * region_width if i < n - 1 else width
#         # 裁剪区域
#         region = image[:, start_col:end_col]
#         # 应用拉普拉斯算子
#         laplace_region = filters.laplace(region)
#         # 计算方差
#         variance = np.var(laplace_region)
#         weight[:, start_col:end_col] *= variance
#         # 打印当前区域的模糊程度评估结果
#         print(f"区域 {i + 1} 的方差: {variance:.10f}")


if __name__ == '__main__':
    file_path = 'D:\\holopix_248\\train\\left view_noise(25)'
    image_list = os.listdir(file_path)
    threshold = 0.003
    for image_ in image_list:
        left_image_path = os.path.join(file_path, image_)
        gradient_image = adaptive_threshold_gradient(left_image_path)
        left_calculate_weights(image_, gradient_image, 31, threshold)

        right_image_path = left_image_path.replace('left view_noise(25)', 'right view_noise(25)')
        gradient_image = adaptive_threshold_gradient(right_image_path)
        right_calculate_weights(image_, gradient_image, 31, threshold)
        # input()