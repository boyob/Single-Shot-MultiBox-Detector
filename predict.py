from utils.utilsPredictInterface import PredictInterface
from PIL import Image
import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', type=float, default=0.5, help='confidence')
    parser.add_argument('--model_image_size', type=tuple, default=(300, 300, 3), help='input image size')
    parser.add_argument('--model_path', type=str, default='data/model_data/ssd_weights.pth', help='the path of model')
    parser.add_argument('--classes_path', type=str, default='data/VOCdevkit/voc_classes.txt', help='class path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = getArgs()
    interface = PredictInterface(args)
    image = Image.open(r'1.jpg')
    r_image = interface.detect_image(image)
    r_image.save(r'r1.jpg')
    r_image.show()