from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from Config import Config


def default_loader(path):
    return Image.open(path)


# 左闭右开
def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


# type(image_data) = type(box_data) = numpy.ndarray, type(image_size) = tuple
def means_normalize(image_data, box_data, image_size):
    # 1.图像减均值、变通道
    image_data = np.transpose(image_data - Config['MEANS'], (2, 0, 1))  # (w,h,c) -> (c,w,h)
    # 2.标注框归一化
    boxes = np.array(box_data[:, :4], dtype=np.float32)
    boxes[:, 0] = boxes[:, 0] / image_size[0]  # xmin / w
    boxes[:, 1] = boxes[:, 1] / image_size[1]  # ymin / h
    boxes[:, 2] = boxes[:, 2] / image_size[0]  # xmax / w
    boxes[:, 3] = boxes[:, 3] / image_size[1]  # ymax / h
    # 3.保证标注框合理性
    boxes = np.maximum(np.minimum(boxes, 1), 0)
    if ((boxes[:, 3] - boxes[:, 1]) <= 0).any() and ((boxes[:, 2] - boxes[:, 0]) <= 0).any():
        return image_data, np.array([])
    # 4.坐标后加类别
    box_data = np.concatenate([boxes, box_data[:, -1:]], axis=-1)
    return image_data, box_data


def my_transform(image, image_boxes, input_size, jitter=0.1, hsv=(0.1, 1.1, 1.1)):
    w, h = input_size  # 300 x 300
    iw, ih = image.size  # 各种尺寸
    hue, sat, val = hsv
    # 1.在横竖维度缩放图像：缩放(iw, ih)->(nw, nh)
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.25, 2)
    if new_ar < 1:
        nh = int(scale * h)  # 高大于宽，按高缩放
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)  # 宽大于高，按宽缩放
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)
    # 2.把缩放后的图像放到尺寸为net_input_size的新建图像上以确保尺寸正确：平移(dx, dy)
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image
    # 3.左右旋转
    flip = rand() < 0.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # 4.扭曲色域
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.0)  # PIL.Image.Image -> numpy.ndarray
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x) * 255
    # 5.标注框跟随图像做相应变动
    if len(image_boxes) > 0:
        np.random.shuffle(image_boxes)  # 只是打乱多个标注框的顺序，每个标注框的信息顺序不变
        # 5.1 缩放、平移
        image_boxes[:, [0, 2]] = image_boxes[:, [0, 2]] * nw / iw + dx  # 对xmin和xmax进行缩放、平移
        image_boxes[:, [1, 3]] = image_boxes[:, [1, 3]] * nh / ih + dy  # 对ymin和ymax进行缩放、平移
        # 5.2 左右翻转
        if flip:
            image_boxes[:, [0, 2]] = w - image_boxes[:, [2, 0]]
        # 5.3 确保标注框不出界
        image_boxes[:, 0:2][image_boxes[:, 0:2] < 0] = 0  # 设置xmin、ymin最小为0
        image_boxes[:, 2][image_boxes[:, 2] > w] = w  # 设置xmax最大为w
        image_boxes[:, 3][image_boxes[:, 3] > h] = h  # 设置ymax最大为h
        # 5.4 筛选出宽高大于1的标注框
        box_w = image_boxes[:, 2] - image_boxes[:, 0]
        box_h = image_boxes[:, 3] - image_boxes[:, 1]
        image_boxes = image_boxes[np.logical_and(box_w > 1, box_h > 1)]
    # 6.变动后再次检查
    if len(image_boxes) == 0:
        return image_data, np.array([])
    # 7.图像去均值、标注框坐标归一化
    if (image_boxes[:, :4] > 0).any():  # 只要有一个子元素大于0就为True
        image_data, image_boxes = means_normalize(image_data, image_boxes, image.size)
        return image_data, image_boxes
    else:
        return image_data, np.array([])
