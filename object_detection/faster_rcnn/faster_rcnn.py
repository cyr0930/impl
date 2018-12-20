from tensorflow import keras
from object_detection.utils import parse_annotations, get_images, calc_iou
from object_detection.faster_rcnn.losses import total_loss
import numpy as np


def calc_reg(pred):
    pass


def make_true_label(annotation):
    def make_true_label_closure(pred):
        return np.concatenate((v_calc_cls(pred[:3]), calc_reg(pred[3:])))
    return make_true_label_closure


root_dir = '../../data/VOC2007'
ann_path = root_dir + '/Annotations'
img_path = root_dir + '/JPEGImages'
num_of_samples = 1
num_of_class = 20
anchors = [(128, 128), (256, 256), (512, 512)]  # according to paper, ratio matters little
iou_threshold = 0.7

annotations = parse_annotations(ann_path, num_of_samples)
images = get_images(img_path, annotations, keras.applications.vgg16.preprocess_input)

base_model = keras.applications.VGG16(weights='imagenet', include_top=False)
intermediate_layer = keras.layers.Conv2D(512, (3, 3), activation='relu')(base_model.output)
cls_output_len = 1*len(anchors)
reg_output_len = 4*len(anchors)
cls_layer = keras.layers.Conv2D(cls_output_len, (1, 1), activation='sigmoid')(intermediate_layer)
reg_layer = keras.layers.Conv2D(reg_output_len, (1, 1))(intermediate_layer)
model = keras.models.Model(inputs=base_model.input, outputs=keras.layers.concatenate([cls_layer, reg_layer]))

model.compile(optimizer='adam', loss=total_loss(cls_output_len + reg_output_len))

v_calc_cls = np.vectorize(lambda a, b: 1. if calc_iou(a, b) >= iou_threshold else 0.)
epoch = 5
for i in range(epoch):
    for key, value in images.items():
        # receptive field of VGG16 is 228
        y_preds = model.predict(value)
        y_true = np.apply_along_axis(make_true_label(annotations[key]), 3, y_preds)
        losses = model.train_on_batch(value, y_true)
        print(losses)
