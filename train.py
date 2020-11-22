import numpy as np
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from model.loss import MultiBoxLoss
from model.net import Net
from Config import Config
from model.dataLoader_utils import my_transform, default_loader
from model.dataLoader import MyDataSet


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--Epoch', type=int, default=50, help='Epoch')
    parser.add_argument('--annotation_path', type=str, default='asset/VOCdevkit/VOC2007_train.txt')
    parser.add_argument('--weight_path', type=str, default='asset/model_data/ssd_weights.pth')
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, lr, gamma, step):
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    # 1.获取命令行参数、创建网络、加载网络参数
    args = getArgs()
    model = Net('train')
    print('-- Loading weights into state dict...')
    pretrained_dict = torch.load(args.weight_path,
                                 map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('-- Loading weights finished.')
    # 2.多GPU并行
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()
    # 3.创建计算loss的类
    criterion = MultiBoxLoss()
    # 4.创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    # 5.读取数据开始训练Epoch轮
    for epoch in range(args.Epoch):
        # 5.1每轮使用不同学习率
        if epoch % 10 == 0:
            adjust_learning_rate(optimizer, args.lr, 0.95, epoch)
        # 5.2创建数据加载器
        train_data = MyDataSet(args.annotation_path, Config['input_size'],
                               transform=my_transform, loader=default_loader)
        # 因为每张图像上的目标个数不确定，所以batch_size只能为1。DataLoader自动把np.array转换成tensor
        data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)
        # 5.3分批训练
        loc_loss = 0
        conf_loss = 0
        batch_image = torch.tensor([], dtype=torch.double)
        batch_boxes = []
        epoch_size = len(data_loader) // args.Batch_size
        for iteration, (image, boxes) in enumerate(data_loader):
            batch_image = torch.cat((batch_image, image), 0)
            batch_boxes.append(boxes.squeeze(0))
            if len(batch_image) == args.Batch_size:
                # 5.3.1获取一批数据
                with torch.no_grad():
                    if torch.cuda.is_available():
                        batch_image = batch_image.cuda().type(torch.FloatTensor)
                        batch_boxes = [ann.cuda().type(torch.FloatTensor) for ann in batch_boxes]
                    else:
                        batch_image = batch_image.type(torch.FloatTensor)
                        batch_boxes = [ann.type(torch.FloatTensor) for ann in batch_boxes]
                # 5.3.2前向传播、计算loss、反向传播
                out = model(batch_image)
                loss_l, loss_c = criterion(out, batch_boxes)
                loss = loss_l + loss_c
                optimizer.zero_grad()  # 梯度置零（否则会与下面计算的梯度累加）
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数
                # 5.3.3统计损失
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()
                print('\nEpoch:' + str(epoch + 1) + '/' + str(args.Epoch))
                print('iter:' + str(iteration) + '/' + str(epoch_size) +
                      ', Loc_Loss: %.4f, Conf_Loss: %.4f.' % (loc_loss / (iteration + 1), conf_loss / (iteration + 1)))
                batch_image = torch.tensor([], dtype=torch.double)
                batch_boxes.clear()
                exit(0)
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), 'logs/Epoch%d-Loc%.4f-Conf%.4f.pth' %
                   ((epoch + 1), loc_loss / (iteration + 1), conf_loss / (iteration + 1)))
