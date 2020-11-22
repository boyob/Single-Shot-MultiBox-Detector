import torch


def encode(matched, defaults, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - defaults[:, :2]
    g_cxcy /= (variances[0] * defaults[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / defaults[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


# (cx, cy, w, h) -> (cx - w/2, cy - h/2, cx + w/2, cy + h/2) = (xmin, ymin, xmax, ymax)
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


# 求重合面积
# box_a.shape=[n,4] box_b.shape=[8732,4] n:一副图像上标注的框的个数
# min_xy=torch.max(ta,tb) min_xy.shape=ta.shape=tb.shape=[n,8732,2]
# return: shape=[n,8732]
#         [[r1, r2, ..., r8732], ...]
#         r1: 第一个标注框与第1个默认框的重合面积，r2: 第一个标注框与第2个默认框的重合面积
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    # unsqueeze和expand:
    # n个标注框中的每一个标注框都要和8732个默认框求重合面积，所以把每个标注框扩展成8732个就可以一一对应地求重合面积了
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # 把相减得到的tensor的小于0的元素换成0
    inter = torch.clamp((max_xy - min_xy), min=0)
    # 计算n个默认框和8732个标注框的重合面积
    return inter[:, :, 0] * inter[:, :, 1]


# 求面积交并比
# return: shape=[n,8732]
#         [[r1_1, r1_2, ..., r1_8732], ..., [rn_1, rn_2, ..., rn_8732]]
#         r1_1: 第1个标注框与第1个默认框的面积交并比，rn_2: 第n个标注框与第2个默认框的面积交并比
def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    # 计算默认框和标注框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def match(idx, gtLocation, gtCategory, defaults, bestGtLocationBS, bestGtCategoryBS, threshold=0.5, variances=None):
    if variances is None:
        variances = [0.1, 0.2]
    overlaps = jaccard(gtLocation, point_form(defaults))
    # 标注框的最好默认框
    _, bestDefaultIdx_of_gt = overlaps.max(1, keepdim=True)
    bestDefaultIdx_of_gt.squeeze_(1)
    # 默认框的最好重合度和最好标注框
    bestOverlap_of_default, bestGtIdx_of_default = overlaps.max(0, keepdim=True)
    bestOverlap_of_default.squeeze_(0)
    bestGtIdx_of_default.squeeze_(0)
    # 重点默认框（该默认框是某个标注框的最好默认框）
    # 1.为重点默认框设置最大的最好重合度(2)
    bestOverlap_of_default.index_fill_(0, bestDefaultIdx_of_gt, 2)
    # 2.为重点默认框分配合适的标注框
    for j in range(bestDefaultIdx_of_gt.size(0)):
        bestGtIdx_of_default[bestDefaultIdx_of_gt[j]] = j
    # 默认框的最好标注框坐标
    bestGtLocation_of_default = gtLocation[bestGtIdx_of_default]
    bestGtLocation_of_default = encode(bestGtLocation_of_default, defaults, variances)
    # 默认框的最好标注框类别
    bestGtCategory_of_default = gtCategory[bestGtIdx_of_default] + 1
    bestGtCategory_of_default[bestOverlap_of_default < threshold] = 0
    # as one in batch_size
    bestGtLocationBS[idx] = bestGtLocation_of_default
    bestGtCategoryBS[idx] = bestGtCategory_of_default
