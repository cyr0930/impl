from xml.etree.ElementTree import parse
import os
import tensorflow as tf
import numpy as np


def parse_annotations(path, num_of_samples=-1, training=True):
    minWidth = 1000
    minHeight = 1000
    annotations = {}
    for fileName in os.listdir(path):
        note = parse(path + '/' + fileName).getroot()
        size = note.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if width < 224 or height < 224:
            # upsample...
            continue
        annotation = [(o.find('name').text, [int(b.text) for b in o.find('bndbox').findall('*')])
                      for o in note.findall('object')
                      if o.find('difficult').text == '0' and (not training or o.find('truncated').text == '0')]
        if len(annotation) > 0:
            annotations[fileName.split('.')[0]] = annotation
        if num_of_samples != -1 and num_of_samples == len(annotations):
            break
    return annotations


def get_images(path, annotations):
    images = []
    for file_name in annotations.keys():
        img_path = path + '/' + file_name + '.jpg'
        img = tf.keras.preprocessing.image.load_img(img_path)
        x = tf.keras.preprocessing.image.img_to_array(img)
        images.append(x)
    return np.array(images)
