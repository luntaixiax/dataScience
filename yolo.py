import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import pandas as pd
from PIL import Image

config = [
    # kernel size x num filters x stride x padding
    (7,64,2,3),
    "M",  # max pool
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
]

def createConvLayers(config, in_channels):
    layers = []
    in_channels = in_channels

    for c in config:
        if isinstance(c, tuple):
            kernel_size, num_filters, stride, padding = c
            layers.append(
                CNNBlock(in_channels, out_channels = num_filters, kernel_size = kernel_size, stride = stride, padding = padding)
            )
            in_channels = num_filters
        elif isinstance(c, str):
            layers.append(
                nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
        elif isinstance(c, list):
            conv1, conv2, num_repeats = c
            kernel_size1, num_filters1, stride1, padding1 = conv1
            kernel_size2, num_filters2, stride2, padding2 = conv2
            for i in range(num_repeats):
                layers.append(
                    CNNBlock(in_channels, out_channels = num_filters1, kernel_size = kernel_size1, stride = stride1,
                             padding = padding1)
                )
                layers.append(
                    CNNBlock(in_channels = num_filters1, out_channels = num_filters2, kernel_size = kernel_size2, stride = stride2,
                             padding = padding2)
                )

                in_channels = num_filters2

    return nn.Sequential(*layers)

def createFcLayers(split_size, num_boxes, num_classes):
    # spit size (s): the original graph will be splitted into s x s grids
    # predicted output: (s x s x pred_dim)
    # where pred_dim = (num_classes + num_boxes * 5 (1 probability that there is an object in box + 4 size/pos)
    # pred_dim: [c1,c2,c3,...., cK, Pb1,x1,y1,w1,h1, Pb2,x2,y2,w2,h2,.....]
    # ck = prob(item belongs to class k), Pbi = prob(there is object inside box i)
    # xi, yi, vector from midpoint of box to left top corner of this grid
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 1024 * split_size * split_size, out_features = 496),
        nn.Dropout(0.0),
        nn.LeakyReLU(0.1),
        nn.Linear(in_features = 496, out_features = split_size * split_size * (num_classes + num_boxes * 5))
    )

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kws):
        super(CNNBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, bias = False, **kws),
            nn.BatchNorm2d(num_features = out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.layer(x)


class Yolov1(nn.Module):
    def __init__(self, in_channels = 3, **kws):
        super(Yolov1, self).__init__()

        self.in_channels = in_channels
        self.darknet = createConvLayers(config, in_channels = in_channels)
        self.fcs = createFcLayers(**kws)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim = 1))


# loss function


# x = torch.randn((2,3,448,448))
# f = Yolov1(in_channels = 3, split_size = 7, num_boxes = 2, num_classes = 20)
# print(f(x).shape)

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()

        self.mse = nn.MSELoss(reduction = "sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5 # weight
        self.lambda_coord = 5  # weight

    def forward(self, predictions, target):
        # assume only 2 boxes, 20 classes to predict
        # N x S x S x 25
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        # ious_maxes, bestbox: N x S x S x 4 , where 4 is (x,y,w,h)
        ious_maxes, bestbox = torch.max(ious, dim = 0) # bestbox {0,1} is find which box (of 2) is better

        # target[..., 20] = whether it has or not has an object(part) in this grid
        # exists_box: N x S x S x 20
        exists_box = target[..., 20].unsqueeze(3)  # identity_obj_i {1,0}

        '''for box coordinates'''
        # box_predictions, box_targets: N x S x S x 4 , where 4 is (x,y,w,h)
        box_predictions = exists_box * (
            (
                # bestbox {0,1}, bestbox = 1: box 1 is better, so use predictions by box1
                bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
            )
        )
        box_targets = exists_box * target[..., 21:25]

        # box_predictions, box_targets: N x S x S x 4 , where 4 is (x,y,w,h)
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # N x S x S x 4 -> N*S*S x 4
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        '''for object loss'''
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        ) # N*S*S x 1
        object_loss = self.mse(
            # N*S*S x 1 -> N*S*S
            torch.flatten(exists_box * pred_box),  # N*S*S
            torch.flatten(exists_box * target[..., 20:21]) # N*S*S
        )

        '''for no object loss'''
        no_object_loss1 = self.mse(
            # N x S x S x 1 -> N x S*S
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1),
        )
        no_object_loss2 = self.mse(
            # N x S x S x 1 -> N x S*S
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 25:26], start_dim = 1),
        )
        no_object_loss = no_object_loss1 + no_object_loss2

        '''for class loss'''
        class_loss = self.mse(
            # N x S x S x 20 -> N*S*S x 20
            torch.flatten(exists_box * predictions[..., :20], end_dim = -2),
            torch.flatten(exists_box * target[..., :20], end_dim = -2),
        )

        '''Total loss'''
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

# prepare data
class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
    
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
    
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
    
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix