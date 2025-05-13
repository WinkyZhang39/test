# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
# from data_set import get_datasets
# from model2_resnet import ClassificationModelResNet
# from torch.utils.data import Dataset, DataLoader


# # 训练模型
# def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
#     # 打印开始训练的信息
#     print('Start Training')
#     # 检查是否有GPU可用
#     print("cuda ?",torch.cuda.is_available())
#     # 设置设备为GPU（如果有）否则为CPU
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # 将模型移动到指定设备
#     model = model.to(device)
#     # 创建TensorBoard的SummaryWriter
#     writer = SummaryWriter()

#     # 遍历每个epoch
#     for epoch in range(num_epochs):
#         # 打印当前epoch
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # 遍历训练和验证阶段
#         for phase in ['train', 'val']:
#             # 如果是训练阶段，将模型设置为训练模式
#             if phase == 'train':
#                 model.train()
#             else:
#                 # 否则设置为评估模式
#                 model.eval()

#             # 初始化损失和正确预测的累积值
#             running_loss = 0.0
#             running_corrects = 0

#             # 遍历当前阶段的数据
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#             writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
#             writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

#     writer.close()
#     return model

# # 主函数
# if __name__ == "__main__":
#     DATA_PATH = 'data'
#     batch_size = 32
#     num_classes = 2
#     num_epochs = 100

#     # 定义数据预处理
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     val_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # 获取数据集
#     train_set, val_set = get_datasets(DATA_PATH, train_transform, val_transform)

#     # 创建DataLoader
#     train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, shuffle=True)

#     # 初始化模型、损失函数和优化器
#     model = ClassificationModelResNet(num_classes=num_classes)
#     criterion = nn.CrossEntropyLoss()#损失函数
#     optimizer = optim.Adam(model.parameters(), lr=0.001)#优化器
#     scheduler = StepLR(optimizer, step_size=7, gamma=0.1)#学习率调度器

#     # 训练模型
#     dataloaders = {'train': train_loader, 'val': val_loader}
#     dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
#     model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs)

#     # 保存模型
#     torch.save(model.state_dict(), 'resnet18_medical_image_classification.pth')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 修改学习率调度器
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from data_set import get_datasets
from model2_resnet import ClassificationModelResNet
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score  # 添加敏感度与AUC计算

# 训练模型（添加早停策略和多指标监控）
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    print('Start Training')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    writer = SummaryWriter()

    # 早停策略参数
    best_loss = float('inf')
    patience = 5
    no_improve_epochs = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每轮epoch记录所有预测结果和标签
        epoch_labels = {'train': [], 'val': []}
        epoch_preds = {'train': [], 'val': []}

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
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
                
                # 收集预测结果用于计算敏感度/AUC
                epoch_labels[phase].extend(labels.cpu().numpy())
                epoch_preds[phase].extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 计算敏感度（召回率）和AUC
            if phase == 'val':
                sensitivity = recall_score(epoch_labels['val'], epoch_preds['val'])
                auc = roc_auc_score(epoch_labels['val'], epoch_preds['val'])
                writer.add_scalar('val/sensitivity', sensitivity, epoch)
                writer.add_scalar('val/auc', auc, epoch)
                print(f'Val Sens: {sensitivity:.4f} AUC: {auc:.4f}')

                # 更新学习率调度器（基于验证损失）
                scheduler.step(epoch_loss)  # 使用ReduceLROnPlateau需要传递指标

            # 修改早停逻辑（基于验证损失+增加耐心值）
            best_auc = 0.0
            patience = 12  # 从5增加到12
            no_improve_epochs = 0

            # 在验证阶段替换原有逻辑
            if phase == 'val':
                current_auc = roc_auc_score(epoch_labels['val'], epoch_preds['val'])
                if current_auc > best_auc:
                    best_auc = current_auc
                    no_improve_epochs = 0
                    best_model_weights = model.state_dict().copy()
                else:
                    no_improve_epochs += 1

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

    # 训练结束后仍加载最佳模型
    model.load_state_dict(best_model_weights)
    writer.close()
    return model

# 主函数改进（数据增强优化与优化器调整）
if __name__ == "__main__":
    DATA_PATH = 'data'
    batch_size = 32
    num_classes = 2
    num_epochs = 100

    # 改进的医学图像增强策略（添加对比度调整与更大旋转角度）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # 缩小裁剪范围
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 仅微调亮度/对比度
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # val_transform修改为确定性预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 获取数据集
    train_set, val_set = get_datasets(DATA_PATH, train_transform, val_transform)

    # 创建DataLoader（调整num_workers避免死锁）
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, 
                             shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4,
                           pin_memory=True)

    # 初始化模型、损失函数和优化器（分层学习率设置）
    model = ClassificationModelResNet(num_classes=num_classes)
    
    # 为不同层设置不同学习率（全连接层更高）
    optimizer = optim.AdamW([
        {'params': model.resnet.fc.parameters(), 'lr': 3e-4},
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-5)

    # 使用ReduceLROnPlateau代替StepLR
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 计算逆频率权重（假设正样本占比60%）
    pos_weight = 0.4 / 0.6  # 负样本比例 / 正样本比例 = 0.4/0.6 ≈ 0.666
    class_weights = torch.tensor([1.0, pos_weight]).to(device)  # 正样本权重降低
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 训练模型
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    model = train_model(model, criterion, optimizer, scheduler, 
                       dataloaders, dataset_sizes, num_epochs)

    # 保存最佳模型
    torch.save(model.state_dict(), 'resnet50_medical_image_classification_best.pth')