from xml.etree.ElementTree import parse
import os
import tensorflow as tf
import numpy as np


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


def calc_iou(a, b):
    pass
