import numpy as np
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from model.SSDDataLoader import DataLoader
from model.SSDLoss import MultiBoxLoss
from model.SSDNet import Net
from utils.utilsConfig import Config


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--Epoch', type=int, default=50, help='Epoch')
    parser.add_argument('--annotation_path', type=str, default='data/VOCdevkit/VOC2007_train.txt', help='the path of annotation')
    parser.add_argument('--weight_path', type=str, default='data/model_data/ssd_weights.pth', help='pretrained weight path')
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
    # 3.读取训练集
    dataLoader = DataLoader(args.Batch_size, (Config["min_dim"], Config["min_dim"]), Config["num_classes"],
                            args.annotation_path)
    dataItem = dataLoader.feed()
    # 4.创建计算loss的类
    criterion = MultiBoxLoss()
    # 5.训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    epoch_size = dataLoader.dataTotal // args.Batch_size
    for epoch in range(args.Epoch):
        if epoch % 10 == 0:
            adjust_learning_rate(optimizer, args.lr, 0.95, epoch)
        loc_loss = 0
        conf_loss = 0
        for iteration in range(epoch_size):
            # 5.1加载数据
            images, targets = next(dataItem)
            with torch.no_grad():
                if torch.cuda.is_available():
                    images = Variable(torch.from_numpy(images).cuda().type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).cuda().type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            # 5.2前向传播、计算loss、反向传播
            out = model(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            optimizer.zero_grad()  # 梯度置零（否则会与下面计算的梯度累加）
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            # 统计损失
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            print('\nEpoch:' + str(epoch + 1) + '/' + str(args.Epoch))
            print('iter:' + str(iteration) + '/' + str(epoch_size) +
                  ', Loc_Loss: %.4f, Conf_Loss: %.4f.' % (loc_loss / (iteration + 1), conf_loss / (iteration + 1)))
            exit(0)
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(),
                   'logs/Epoch%d-Loc%.4f-Conf%.4f.pth' % (
                   (epoch + 1), loc_loss / (iteration + 1), conf_loss / (iteration + 1)))
