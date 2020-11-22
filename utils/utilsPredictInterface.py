import numpy as np
import colorsys
import os
import torch
from model.SSDNet import Net
import torch.backends.cudnn as cudnn
# from utils.utilsPredict import letterbox_image,ssd_correct_boxes
from PIL import ImageFont, ImageDraw
from torch.autograd import Variable
import numpy as np
from PIL import Image

MEANS = (104, 117, 123)


# 把图像缩放到指定尺寸。因为要保持纵横比，所以可能需要填充灰色
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


# top, left, bottom, right的shape = (n, 1)，其中n是检测到的目标个数。
# input_shape: 网络接收的尺寸(300, 300)。
# image_shape: 原始图像的尺寸(高, 宽)。
# new_shape: 原始图像宽高等比例缩放到最大边长为300后的尺寸。
# 如果原始图像宽高不相等，缩放后就有一个边长为300，另一个边长小于300。
# 针对这种情况，前面的做法是在两边（上下或左右）填充灰色，构成一个宽高都为300的图像做为输入。把这个图像称为IMG。
# top, left, bottom, right是在把IMG的边长看为1时，目标外接框的边界坐标。
# return的boxes是原图上目标的边界坐标。boxes.shape = (n, 4)。
def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = np.max(image_shape)
    top = (top - offset[0]) * scale
    left = (left - offset[1]) * scale
    bottom = (bottom - offset[0]) * scale
    right = (right - offset[1]) * scale
    boxes = np.concatenate([top, left, bottom, right], axis=-1)
    return boxes


class PredictInterface:
    def __init__(self, args):
        self.args = args
        self.class_names = self.get_class()
        self.generate()

    def get_class(self):
        with open(self.args.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        self.num_classes = len(self.class_names) + 1
        model = Net("test")
        self.net = model
        print('-- Loading model from {} ...'.format(self.args.model_path))
        model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
        self.net = torch.nn.DataParallel(self.net)
        cudnn.benchmark = True
        # self.net = self.net.cuda()
        # 为每个类别的外接框设置不同颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        crop_img = np.array(letterbox_image(image, (self.args.model_image_size[0], self.args.model_image_size[1])))
        # 图片预处理，归一化
        photo = Variable(
            torch.from_numpy(np.expand_dims(np.transpose(crop_img - MEANS, (2, 0, 1)), 0)).type(torch.FloatTensor))
        preds = self.net(photo)
        top_conf = []
        top_label = []
        top_bboxes = []
        for i in range(preds.size(1)):
            j = 0
            while preds[0, i, j, 0] >= self.args.confidence:
                score = preds[0, i, j, 0]
                label_name = self.class_names[i - 1]
                pt = (preds[0, i, j, 1:]).detach().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                top_conf.append(score)
                top_label.append(label_name)
                top_bboxes.append(coords)
                j = j + 1
        # 将预测结果进行解码
        if len(top_conf) <= 0:
            return image
        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = \
            np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), \
            np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
        # 去掉灰条
        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                  np.array([self.args.model_image_size[0], self.args.model_image_size[1]]), image_shape)
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.args.model_image_size[0]
        for i, c in enumerate(top_label):
            predicted_class = c
            score = top_conf[i]
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
            # 画框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image