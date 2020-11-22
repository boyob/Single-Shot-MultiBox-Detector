from random import shuffle
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image

MEANS = (104, 117, 123)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class DataLoader(object):
    def __init__(self, batch_size, image_size, num_classes, annotation_path):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes - 1
        self.annotation_path = annotation_path
        self.load_annotation()

    def load_annotation(self):
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        self.train_lines = lines
        self.dataTotal = len(lines)

    # 处理一张图像和它的标注信息
    def get_random_data(self, annotation_line, input_shape, jitter=.1, hue=.1, sat=1.1, val=1.1):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        # box = [[xmin1 ymin1 xmax1 ymax1 class1] ... [xminn yminn xmaxn ymaxn classn]] （n是标注目标的个数, x横向，y纵向）
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x) * 255  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def feed(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            for annotation_line in lines:
                img, y = self.get_random_data(annotation_line, self.image_size[0:2])
                if len(y) == 0:  # 没有标注信息
                    continue
                boxes = np.array(y[:, :4], dtype=np.float32)
                boxes[:, 0] = boxes[:, 0] / self.image_size[1]
                boxes[:, 1] = boxes[:, 1] / self.image_size[0]
                boxes[:, 2] = boxes[:, 2] / self.image_size[1]
                boxes[:, 3] = boxes[:, 3] / self.image_size[0]
                boxes = np.maximum(np.minimum(boxes, 1), 0)
                if ((boxes[:, 3] - boxes[:, 1]) <= 0).any() and ((boxes[:, 2] - boxes[:, 0]) <= 0).any():
                    continue
                y = np.concatenate([boxes, y[:, -1:]], axis=-1)
                inputs.append(np.transpose(img - MEANS, (2, 0, 1)))
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets
