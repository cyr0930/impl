import tensorflow as tf
import numpy as np
from object_detection.utils import parse_annotations, get_images

root_dir = '../../data/VOC2007'
ann_path = root_dir + '/Annotations'
img_path = root_dir + '/JPEGImages'
num_of_samples = -1
num_of_class = 20

annotations = parse_annotations(ann_path, num_of_samples)
images = get_images(img_path, annotations)
images = images / 128. - 1.     # preprocessing

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_of_class, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# freeze
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit_generator(...)    # Not Implemented

# not freeze
for layer in model.layers[19:]:
   layer.trainable = True
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit_generator(...)    # Not Implemented
