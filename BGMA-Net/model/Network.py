import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.BGMA import BGMA

#下面网络中用到的卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

#注意力机制门
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_x = nn.Conv2d(in_channels, inter_channels, stride=1,kernel_size=1)
        self.W_g = nn.Conv2d(gating_channels, inter_channels,stride=1, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: 输入特征图, g: 门控信号（权重图）
        gate = self.W_g(g)
        x_conv = self.W_x(x)
        # print(f"x.shape: {x.shape}")  # 输入特征图的尺寸
        # print(f"g.shape: {g.shape}")  # 输入权重图的尺寸
        # print(f"x_conv.shape: {x_conv.shape}")  # 经过 W_x 的输出尺寸
        # print(f"gate.shape: {gate.shape}")  # 经过 W_g 的输出尺寸
        weight_1 = self.relu(x_conv + gate)
        weight_2 = self.psi(weight_1)
        return x * weight_2


#BGMA-net(Trainable params: 4,059,992)
class BGMA_net(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, width=16):
        super(BGMA_net, self).__init__()
        self.conv1 = ConvBlock(n_channel_in, width)
        self.conv2 = ConvBlock(width, width * 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(width * 2, width * 4)
        self.conv4 = ConvBlock(width * 4, width * 8)
        self.conv5 = ConvBlock(width * 8, width * 8)
        self.conv6 = ConvBlock(width * 8, width * 8)
        self.conv7 = ConvBlock(width * 16, width * 16)
        self.conv8 = ConvBlock(width * 16, width * 8)
        self.conv9 = ConvBlock(width * 10, width * 8)
        self.conv10 = ConvBlock(width * 8, width * 4)

        self.att1 = BGMA(in_channels=width * 8, num_heads=8, head_dim=32,kernel_size_o= 3)
        self.att2 = BGMA(in_channels=width * 2, num_heads=8, head_dim=32,kernel_size_o= 5)
        self.att3 = BGMA(in_channels=1, num_heads=4, head_dim=16, kernel_size_o= 9)

        self.up1 = nn.ConvTranspose2d(width * 8, width * 8, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(width * 8, width * 8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(width * 4, width * 4, kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(in_channels=width * 4 + 1, out_channels=width * 2, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels=width * 2, out_channels=width , kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels=width , out_channels=1, kernel_size=1)

    def forward(self, x, weight_map):
        #只有测试时运用
        # x = x.float()
        # if x.ndimension() == 3:  # 如果是3D张量，例如 [32, 248, 248]
        #     x = x.unsqueeze(1)  # 在第1维增加一个维度，变成 [32, 1, 248, 248]
        c1 = self.conv1(x)
        f1 = self.conv2(c1)
        f2 = self.pool1(f1)
        c2 = self.conv3(f2)
        f3 = self.conv4(c2)
        f4 = self.pool1(f3)
        c3 = self.conv5(f4)
        f5 = self.conv6(c3)
        f6 = self.pool1(f5)
        c4 = self.conv5(f6)
        f7 = self.conv6(c4)

        m1 = self.conv1(weight_map)
        weight_map1 = self.conv2(m1)
        weight_map2 = self.pool1(weight_map1)
        m2 = self.conv3(weight_map2)
        weight_map3 = self.conv4(m2)
        weight_map4 = self.pool1(weight_map3)

        f8 = self.up1(f7)
        f_4 = self.att1(f4,weight_map4)
        merge1 = torch.cat([f8, f_4], dim=1)
        c5 = self.conv7(merge1)
        f9 = self.conv8(c5)
        f10= self.up2(f9)
        f_2 = self.att2(f2, weight_map2)
        merge2 = torch.cat([f10, f_2], dim=1)
        c6 = self.conv9(merge2)
        f11 = self.conv10(c6)
        f12 = self.up3(f11)
        f_1 = self.att3(x, weight_map)
        merge3 = torch.cat([f12, f_1], dim=1)
        f13 = self.conv11(merge3)
        f14 = self.conv12(f13)
        output = self.conv13(f14)

        return output



class Unet_double(nn.Module):
    def __init__(self):
        super(Unet_double, self).__init__()
        self.Unet_left = Attention_Unet(n_channel_in=1, n_channel_out=1, width=16)
        self.Unet_right = Attention_Unet(n_channel_in=1, n_channel_out=1, width=16)

    def forward(self, x_l, weight_l, x_r, weight_r):
        clear_l = self.Unet_left(x_l, weight_l)
        clear_r = self.Unet_right(x_r, weight_r)

        return clear_l, clear_r


class BGMA_net_double(nn.Module):
    def __init__(self):
        super(BGMA_net_double, self).__init__()
        self.Unet_left = BGMA_net(n_channel_in=1, n_channel_out=1, width=16)
        self.Unet_right = BGMA_net(n_channel_in=1, n_channel_out=1, width=16)

    def forward(self, x_l, weight_l, x_r, weight_r):
        clear_l = self.Unet_left(x_l, weight_l)
        clear_r = self.Unet_right(x_r, weight_r)

        return clear_l, clear_r

if __name__ == "__main__":
    from torchsummary import summary
    from torchvision import transforms
    from PIL import Image

    # 确定设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取图像
    image_path = "D:/LFDOF/LFDOF/train_data/input/IMG_1275/IMG_1275_ap1_00.png"
    image = Image.open(image_path)

    # 检查图像的模式
    print(f"Original image mode: {image.mode}")

    # 如果图像是 RGBA 模式（即有透明度通道），则转换为 RGB 模式
    if image.mode == 'RGBA':
        image = image.convert('RGB')
        print("Image converted to RGB mode.")

    # 定义图像预处理（转换为 tensor，并归一化到 [0, 1] 范围）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
    ])

    # 应用预处理
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch size 维度并移至设备
    print(input_tensor.shape)  # 查看图像张量的形状，应该是 [1, 3, 688, 1008]

    # 实例化模型并将其移至相同的设备
    model = Unet(n_channel_in=3, n_channel_out=3, width=16).to(device)

    # 进行一次前向传播，测试模型逻辑
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # 输出预测结果的形状


    # 计算模型的参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # 打印模型的总参数量
    print(f"Total trainable parameters: {count_parameters(model)}")

    # 使用 torchsummary 打印模型的概述
    summary(model, input_size=(3, 1008, 688))  # 输入大小为 (3, 1008, 688)

    # 创建模型实例
    # model = BGMA_net(n_channel_in=1, n_channel_out=1, width=16)
    # model = Unet(n_channel_in=1, n_channel_out=1, width=16)
    #
    # # 选择设备 (GPU/CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    #
    # # 设置模型为评估模式
    # model.eval()

    # # 创建模拟的输入数据
    # image_input = torch.randn(1, 1, 248, 248).to(device)  # 图像输入
    # weight_map_input = torch.randn(1, 1, 248, 248).to(device)  # 权重图输入
    #
    # # 调用模型并打印输出形状
    # output = model(image_input, weight_map_input)
    # print("Output shape:", output.shape)

    # # 创建模拟的输入数据
    # image_input = torch.randn(1, 1, 248, 248)  # 假设输入图像是1通道248x248的图像
    # weight_map_input = torch.randn(1, 1, 248, 248)  # 假设输入权重图也是1通道248x248的图像
    #
    # # 打印模型的参数量
    # # summary(model, input_size=[(1, 248, 248), (1, 248, 248)])  # 输入两个参数，图像和权重图
    # summary(model, input_size=[(1, 248, 248)])  # 输入两个参数，图像和权重图
