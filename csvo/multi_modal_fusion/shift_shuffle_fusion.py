import torch
import torch.nn as nn
import torch.optim as optim
import random
def shuffle(x1, x2):
    # 获取输入的形状，假设输入的 x1 和 x2 形状相同
    batch_size, channels, height, width = x1.size()
    
    # 随机选择 n/2 个通道进行互换
    num_channels_to_swap = channels // 2
    indices_to_swap = random.sample(range(channels), num_channels_to_swap)
    
    # 创建交换后的新张量
    x1_shuffled = x1.clone()
    x2_shuffled = x2.clone()
    
    # 进行通道交换
    x1_shuffled[:, indices_to_swap, :, :] = x2[:, indices_to_swap, :, :]
    x2_shuffled[:, indices_to_swap, :, :] = x1[:, indices_to_swap, :, :]
    
    return x1_shuffled, x2_shuffled
def shift(x):
    # 获取输入的形状
    batch_size, channels, height, width = x.size()
    
    # 随机选择移动方向
    direction = random.choice(['up', 'down', 'left', 'right'])
    
    # 创建一个零填充的张量，保持与输入相同的形状
    shifted_x = torch.zeros_like(x)
    
    if direction == 'up':
        # 向上移动1格
        shifted_x[:, :, :-1, :] = x[:, :, 1:, :]
    elif direction == 'down':
        # 向下移动1格
        shifted_x[:, :, 1:, :] = x[:, :, :-1, :]
    elif direction == 'left':
        # 向左移动1格
        shifted_x[:, :, :, :-1] = x[:, :, :, 1:]
    elif direction == 'right':
        # 向右移动1格
        shifted_x[:, :, :, 1:] = x[:, :, :, :-1]
    
    return shifted_x
class DualModalityFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualModalityFusion, self).__init__()
        # 1x1卷积层 + BatchNorm
        self.conv1x1_mod1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1_mod1 = nn.BatchNorm2d(out_channels)
        
        self.conv1x1_mod2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1_mod2 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积层 + BatchNorm
        self.conv3x3_mod1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2_mod1 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_mod2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2_mod2 = nn.BatchNorm2d(out_channels)
        
        # 1x1卷积层 + BatchNorm (第二次)
        self.conv1x1_mod1_second = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3_mod1 = nn.BatchNorm2d(out_channels)
        
        self.conv1x1_mod2_second = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3_mod2 = nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2):
        # 1x1卷积 + BatchNorm
        out1 = self.bn1_mod1(self.conv1x1_mod1(x1))
        out2 = self.bn1_mod2(self.conv1x1_mod2(x2))
        
        # Shuffle
        out1, out2 = shuffle(out1, out2)
        
        res1 = out1
        res2 = out2
        # Shift
        out1 = shift(out1)
        out2 = shift(out2)
        
        # 3x3卷积 + BatchNorm
        res1 = self.bn2_mod1(self.conv3x3_mod1(res1))
        res2 = self.bn2_mod2(self.conv3x3_mod2(res2))
        # Add操作
        out1 = out2 + res1
        out2 = out1 + res2
        
        # print(out1.shape, out2.shape)
        # Shuffle
        out1, out2 = shuffle(out1, out2)
        
        # 1x1卷积 + BatchNorm (第二次)
        out1 = self.bn3_mod1(self.conv1x1_mod1_second(out1))
        out2 = self.bn3_mod2(self.conv1x1_mod2_second(out2))
        # Residual加法
        out1 = out1 + x1
        out2 = out2 + x2
        
        return out1, out2

# 定义随机目标函数，用于计算损失
def random_target_function(x):
    # 假设目标是输出的形状相同的矩阵，值随机
    return torch.randn_like(x)

# 定义训练过程
def train(model, num_epochs=5, batch_size=8, learning_rate=0.001):
    # 损失函数
    criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 模拟输入数据
    in_channels = 128
    out_channels = 3
    input_height = 160
    input_width = 120
    
    for epoch in range(num_epochs):
        # 每个epoch生成一批新的随机输入
        x1 = torch.randn(batch_size, in_channels, input_height, input_width)
        x2 = torch.randn(batch_size, in_channels, input_height, input_width)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output1, output2 = model(x1, x2)
        
        # 计算目标输出
        target1 = random_target_function(output1)
        target2 = random_target_function(output2)
        
        # 计算损失
        loss1 = criterion(output1, target1)
        loss2 = criterion(output2, target2)
        
        # 总损失
        loss = loss1 + loss2
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 打印当前epoch的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # 模型实例化
    x = torch.tensor([[1,2,3]])
    print(x.shape, x[0].shape)
    in_channels = 128
    out_channels = 128
    model = DualModalityFusion(in_channels, out_channels)

    # 开始训练
    train(model, num_epochs=50, batch_size=8, learning_rate=0.001)