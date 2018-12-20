from xml.etree.ElementTree import parse
import os
import tensorflow as tf
import numpy as np
import math
import random


def parse_annotations(path, num_of_samples=-1, s=600, training=True):
    annotations = {}
    for fileName in os.listdir(path):
        note = parse(path + '/' + fileName).getroot()
        size = note.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        min_size = min(width, height)
        upsampling_ratio = s / min_size if min_size < s else None
        u_func = lambda x: round(x * upsampling_ratio) if upsampling_ratio else x
        annotation = [(o.find('name').text, [u_func(int(b.text)) for b in o.find('bndbox').findall('*')])
                      for o in note.findall('object')
                      if o.find('difficult').text == '0' and (not training or o.find('truncated').text == '0')]
        if len(annotation) > 0:
            annotations[fileName.split('.')[0]] = annotation
        if num_of_samples != -1 and num_of_samples == len(annotations):
            break
    return annotations


def get_images(path, annotations, preprocess_func, s=600):
    images = {}
    for file_name in annotations.keys():
        img_path = path + '/' + file_name + '.jpg'
        img = tf.keras.preprocessing.image.load_img(img_path)
        width, height = img.size
        min_size = min(width, height)
        upsampling_ratio = s / min_size if min_size < s else None
        if upsampling_ratio:
            new_size = tuple(map(lambda x: round(x * upsampling_ratio), img.size))
            img = img.resize(new_size)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = preprocess_func(x)
        x = np.expand_dims(x, axis=0)
        images[file_name] = x
    return images


def make_label(annotation, pred, anchors, true_threshold=0.7, false_threshold=0.3):
    # positive label (1, ground_truth)
    # negative label (0, anything)
    # non label      (cls_pred, reg_pred), leads to 0 loss
    label = np.zeros(pred.shape)
    pos_idx_list = []
    neg_idx_list = []
    for i in range(label.shape[1]):
        cur_x = 113 + 16 * i
        for j in range(label.shape[2]):
            cur_y = 113 + 16 * j
            for anchor_idx in range(len(anchors)):
                anchor_pos = (cur_x, cur_y, anchors[anchor_idx][0], anchors[anchor_idx][1])
                max_iou = 0.
                max_idx = 0
                min_iou = 1.
                for a_idx in range(len(annotation)):
                    iou = calc_iou(anchor_pos, annotation[a_idx][1])
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = a_idx
                    elif iou < min_iou:
                        min_iou = iou
                offset = len(anchors) + anchor_idx * 4
                if max_iou >= true_threshold:
                    pos_idx_list.append((i, j, anchor_idx))
                    label[0][i][j][anchor_idx] = 1.
                    label[0][i][j][offset + 0] = (annotation[max_idx][1][0] - anchor_pos[0]) / anchor_pos[2]
                    label[0][i][j][offset + 1] = (annotation[max_idx][1][1] - anchor_pos[1]) / anchor_pos[3]
                    label[0][i][j][offset + 2] = math.log(annotation[max_idx][1][2] / anchor_pos[2])
                    label[0][i][j][offset + 3] = math.log(annotation[max_idx][1][3] / anchor_pos[3])
                elif min_iou > false_threshold:
                    label[0][i][j][anchor_idx] = pred[0][i][j][anchor_idx]
                    label[0][i][j][offset + 0] = pred[0][i][j][offset + 0]
                    label[0][i][j][offset + 1] = pred[0][i][j][offset + 1]
                    label[0][i][j][offset + 2] = pred[0][i][j][offset + 2]
                    label[0][i][j][offset + 3] = pred[0][i][j][offset + 3]
                else:
                    neg_idx_list.append((i, j, anchor_idx))
    random.shuffle(neg_idx_list)
    if len(pos_idx_list) <= 128:
        num_of_neg_label = 256 - len(pos_idx_list)
        neg_idx_list = neg_idx_list[num_of_neg_label:]
    else:
        random.shuffle(pos_idx_list)
        neg_idx_list = neg_idx_list[128:]
        pos_idx_list = pos_idx_list[128:]
        for pos_idx in pos_idx_list:
            offset = len(anchors) + pos_idx[2] * 4
            label[0][pos_idx[0]][pos_idx[1]][pos_idx[2]] = pred[0][pos_idx[0]][pos_idx[1]][pos_idx[2]]
            label[0][pos_idx[0]][pos_idx[1]][offset + 0] = pred[0][pos_idx[0]][pos_idx[1]][offset + 0]
            label[0][pos_idx[0]][pos_idx[1]][offset + 1] = pred[0][pos_idx[0]][pos_idx[1]][offset + 1]
            label[0][pos_idx[0]][pos_idx[1]][offset + 2] = pred[0][pos_idx[0]][pos_idx[1]][offset + 2]
            label[0][pos_idx[0]][pos_idx[1]][offset + 3] = pred[0][pos_idx[0]][pos_idx[1]][offset + 3]
    for neg_idx in neg_idx_list:
        offset = len(anchors) + neg_idx[2] * 4
        label[0][neg_idx[0]][neg_idx[1]][neg_idx[2]] = pred[0][neg_idx[0]][neg_idx[1]][neg_idx[2]]
        label[0][neg_idx[0]][neg_idx[1]][offset + 0] = pred[0][neg_idx[0]][neg_idx[1]][offset + 0]
        label[0][neg_idx[0]][neg_idx[1]][offset + 1] = pred[0][neg_idx[0]][neg_idx[1]][offset + 1]
        label[0][neg_idx[0]][neg_idx[1]][offset + 2] = pred[0][neg_idx[0]][neg_idx[1]][offset + 2]
        label[0][neg_idx[0]][neg_idx[1]][offset + 3] = pred[0][neg_idx[0]][neg_idx[1]][offset + 3]
    return label


def calc_iou(a, b):
    x_a1 = a[0] - a[2] / 2
    y_a1 = a[1] - a[3] / 2
    x_a2 = a[0] + a[2] / 2
    y_a2 = a[1] + a[3] / 2
    x_b1 = b[0] - b[2] / 2
    y_b1 = b[1] - b[3] / 2
    x_b2 = b[0] + b[2] / 2
    y_b2 = b[1] + b[3] / 2
    inner_area = max(0, min(x_a2, x_b2) - max(x_a1, x_b1) + 1) * max(0, min(y_a2, y_b2) - max(y_a1, y_b1) + 1)
    a_area = a[2] * a[3]
    b_area = b[2] * b[3]
    return inner_area / (a_area + b_area - inner_area)
