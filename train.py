import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from data_set import get_datasets
from model1_resnet import ClassificationModelResNet
from torch.utils.data import Dataset, DataLoader


# 训练模型
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    # 打印开始训练的信息
    print('Start Training')
    # 检查是否有GPU可用
    print("cuda ?",torch.cuda.is_available())
    # 设置设备为GPU（如果有）否则为CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 将模型移动到指定设备
    model = model.to(device)
    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter()

    # 遍历每个epoch
    for epoch in range(num_epochs):
        # 打印当前epoch
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 遍历训练和验证阶段
        for phase in ['train', 'val']:
            # 如果是训练阶段，将模型设置为训练模式
            if phase == 'train':
                model.train()
            else:
                # 否则设置为评估模式
                model.eval()

            # 初始化损失和正确预测的累积值
            running_loss = 0.0
            running_corrects = 0

            # 遍历当前阶段的数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

    writer.close()
    return model

# 主函数
if __name__ == "__main__":
    DATA_PATH = 'data'
    batch_size = 32
    num_classes = 2#分类结果是是或者不是
    num_epochs = 30

    # 定义数据预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 随机改变亮度、对比度、饱和度和色调
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 随机改变亮度、对比度、饱和度和色调
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 获取数据集
    train_set, val_set = get_datasets(DATA_PATH, train_transform, val_transform)

    # 创建DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = ClassificationModelResNet(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs)

    # 保存模型
    torch.save(model.state_dict(), 'resnet18_medical_image_classification.pth')