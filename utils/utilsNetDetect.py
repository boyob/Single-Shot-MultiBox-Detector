import torch
from torch.autograd import Function
from utils.utilsConfig import Config


class Detect(Function):
    @staticmethod
    def forward(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh,
                loc_data, conf_data, prior_data):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']

        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        # 对每一张图片进行处理
        for i in range(num):
            # 对先验框解码获得预测框
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # shape=[8732,4]
            conf_scores = conf_preds[i].clone()  # shape=[21,8732]

            for cl in range(1, self.num_classes):
                # 对每一类进行非极大抑制
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # shape = [8732]
                scores = conf_scores[cl][c_mask]  # shape = [x]: 有x个默认框检测到（类别为cl的）目标
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)  # shape=[8732,4]
                boxes = decoded_boxes[l_mask].view(-1, 4)  # shape = [x,4]: 获取这x个默认框检测到的位置
                # 进行非极大抑制
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


# loc = [g_cx, g_cy, g_w, g_h]
# priors = [d_cx, d_cy, d_w, d_h], 范围：[0, 1]
# variances = [0.1, 0.2]
# boxes = [minx, miny, maxx, maxy]
#       = [d_cx + 0.1 * g_cx * d_w - (d_w * np.exp(0.2 * g_w)) / 2.0,
#          d_cy + 0.1 * g_cy * d_h - (d_h * np.exp(0.2 * g_h)) / 2.0,
#          d_cx + 0.1 * g_cx * d_w + (d_w * np.exp(0.2 * g_w)) / 2.0,
#          d_cy + 0.1 * g_cy * d_h + (d_h * np.exp(0.2 * g_h)) / 2.0]
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# boxes.shape = [x, 4]
# scores.shape = [x]
# https://github.com/fmassa/object-detection.torch
def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]  # 面积最大的top_k个
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:  # 所有元素的个数
        i = idx[-1]
        # 1.分数最高的位置被选中
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        # 2.从剩下的位置中筛选出与刚被选中的位置的交并比小于等于overlap的位置，保留它们的序号在idx中
        idx = idx[:-1]
        # 2.1交集区域的边界坐标
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        # 2.2交集区域的面积
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # 2.3并集区域的面积
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        # 2.4交并比
        IoU = inter / union
        # 2.5选择交并比小于等于overlap的位置，保留它们的序号在idx中
        idx = idx[IoU.le(overlap)]
    return keep, count
