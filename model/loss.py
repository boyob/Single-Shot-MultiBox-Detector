import torch
from Config import Config
from model.loss_match import match


class MultiBoxLoss(torch.nn.Module):
    def __init__(self,
                 overlap_thresh=0.5,
                 prior_for_matching=True,
                 bkg_label=0,
                 neg_mining=True,
                 neg_pos=3,
                 neg_overlap=0.5,
                 encode_target=False):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = Config['num_classes']
        self.variance = Config['variance']
        self.threshold = overlap_thresh
        self.use_prior_for_matching = prior_for_matching
        self.background_label = bkg_label
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.encode_target = encode_target

    def forward(self, net_out, targets):
        location, category, defaultbox = net_out
        batch_size = location.shape[0]
        num_defaultbox = defaultbox.shape[0]
        # 1.找出8732个默认框与它的最好标注框的坐标差，找出8732个默认框的最好标注框的类别
        bestGtLocationBS = torch.Tensor(batch_size, num_defaultbox, 4)
        bestGtCategoryBS = torch.LongTensor(batch_size, num_defaultbox)
        for idx in range(batch_size):
            gtLocation = targets[idx][:, :-1].data
            gtCategory = targets[idx][:, -1].data
            defaults = defaultbox.data
            match(idx, gtLocation, gtCategory, defaults, bestGtLocationBS, bestGtCategoryBS, self.threshold,
                  self.variance)
        if Config['cuda']:
            bestGtLocationBS = bestGtLocationBS.cuda()
            bestGtCategoryBS = bestGtCategoryBS.cuda()
        bestGtLocationBS = torch.autograd.Variable(bestGtLocationBS, requires_grad=False)
        bestGtCategoryBS = torch.autograd.Variable(bestGtCategoryBS, requires_grad=False)
        # 2.标记正样本
        pos = bestGtCategoryBS > 0
        # 3.位置Loss：只计算正样本
        positiveIdx = pos.unsqueeze(pos.dim()).expand_as(location)
        positiveLoc = location[positiveIdx].view(-1, 4)
        bestGtLocationBS = bestGtLocationBS[positiveIdx].view(-1, 4)
        loss_l = torch.nn.functional.smooth_l1_loss(positiveLoc, bestGtLocationBS,
                                                    reduction='sum')  # size_average=False)
        # 4.分类Loss
        loss_c = torch.nn.functional.cross_entropy(category.view(-1, self.num_classes), bestGtCategoryBS.view(-1),
                                                   reduction='none')
        loss_c = loss_c.view(batch_size, -1)
        loss_c[pos] = 0
        # 4.1获得每一张图新的softmax的结果
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 4.2设置负样本数量是正样本的3倍、标记负样本
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # 4.3计算正样本和负样本的分类loss(剩下的样本既不属于正样本也不属于负样本)
        positiveIdx = pos.unsqueeze(2).expand_as(category)
        negativeIdx = neg.unsqueeze(2).expand_as(category)
        lgCategory = category[(positiveIdx + negativeIdx).gt(0)].view(-1, self.num_classes)
        gtCategory = bestGtCategoryBS[(pos + neg).gt(0)]
        loss_c = torch.nn.functional.cross_entropy(lgCategory, gtCategory, reduction='sum')  # size_average=False)
        # 5.后处理
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
