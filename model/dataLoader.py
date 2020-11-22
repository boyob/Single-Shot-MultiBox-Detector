import torch
import numpy as np


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, root, input_size, transform=None, loader=None):
        # 1.创建data_items
        # data_items = [imagePath, [[xmin1, ymin1, xmax1, ymax1, class1], [xmin2, ymin2, xmax2, ymax2, class2], ...]]
        self.data_items = []
        data_lines = open(root, 'r')
        for data_line in data_lines:
            line = data_line.strip().split(' ')
            image_path = line[0]
            image_boxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
            self.data_items.append([image_path, image_boxes])
        # 2.初始化一些成员变量
        self.input_size = input_size
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # 1.读取图像和标注坐标
        image_path, image_boxes = self.data_items[index]
        image = self.loader(image_path)
        # 2.图像变换、标注坐标变换、图像类型由PIL.Image转np.array
        if self.transform is not None:
            image, image_boxes = self.transform(image, image_boxes, self.input_size)
        else:
            image = np.array(image, dtype=float)
        return image, image_boxes

    def __len__(self):
        return len(self.data_items)


if __name__ == '__main__':
    from PIL import Image
    from Config import Config
    from model.dataLoader_utils import my_transform, default_loader

    for epoch in range(5):  # 遍历5次训练数据
        # 1.每次遍历都要重新创建数据加载器
        train_data = MyDataSet(r'E:\projects\SSD\asset\train.txt', Config['input_size'],
                               transform=my_transform, loader=default_loader)
        # 因为每张图像上的目标个数不确定，所以batch_size只能为1。DataLoader自动把np.array转换成tensor
        data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)
        # 2.遍历
        batch_image = []
        batch_boxes = []
        for image, boxes in data_loader:  # 遍历1次训练数据
            batch_image.append(image)  # image.shape=[b, c, h, w]
            batch_boxes.append(boxes)
            if len(batch_image) == 2:  # batch_size
                # 使用batch_size数据
                for i, (image, boxes) in enumerate(zip(batch_image, batch_boxes)):
                    # 保存变换后的图像
                    image_hwc = image.squeeze().permute(1, 2, 0).numpy()
                    im = Image.fromarray(np.uint8(image_hwc))
                    im.save('E:\\projects\\SSD\\asset\\z_' + str(epoch) + '_' + str(i) + '.jpg')
                    # 输出变换后的框坐标
                    print(boxes)
                batch_image.clear()
                batch_boxes.clear()
