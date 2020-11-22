import argparse
import xml.etree.ElementTree as ET
import os

"""
1. 从VOC_path/VOC2007/ImageSets/Main读取：train.txt、val.txt、test.txt，
   从VOC_path/VOC2007/Annotations读取xml文件，
   把结果存放在该脚本所在文件夹：VOC2007_train.txt、VOC2007_val.txt、VOC2007_test.txt。
2. 存放形式：...\VOCdevkit\VOC2007\JPEGImages\XXXXXX.jpg xmin1,ymin1,xmax1,ymax1,class1 xmin2,ymin2,xmax2,ymax2,class2 ...
   x:从左向右 y:从上向下
"""


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--VOC_path', type=str, default=r'E:\boyob\Projects\VOCdevkit', help='the path of VOC dataset')
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help='the path to save data item file')
    args = parser.parse_args()
    return args


def convert_annotation(list_file, xml_path):
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':
    args = getArgs()
    sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    for year, image_set in sets:
        txt_path = os.path.join(args.VOC_path, 'VOC%s' % year, 'ImageSets', 'Main', '%s.txt' % image_set)
        save_path = os.path.join(args.save_path, 'VOC%s_%s.txt' % (year, image_set))
        image_ids = open(txt_path).read().strip().split()
        list_file = open(save_path, 'w')
        for image_id in image_ids:
            dataItem = os.path.join(args.VOC_path, 'VOC%s' % year, 'JPEGImages', '%s.jpg' % image_id)
            list_file.write(dataItem)
            xml_path = os.path.join(args.VOC_path, 'VOC%s' % year, 'Annotations', '%s.xml' % image_id)
            convert_annotation(list_file, xml_path)
            list_file.write('\n')
        list_file.close()
