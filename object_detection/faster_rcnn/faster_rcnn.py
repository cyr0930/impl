import tensorflow as tf
from tensorflow import keras
from object_detection.faster_rcnn.utils import parse_annotations, get_images, make_label
from object_detection.faster_rcnn.losses import total_loss
import numpy as np
import math

root_dir = '../../data/VOC2007'
ann_path = root_dir + '/Annotations'
img_path = root_dir + '/JPEGImages'
num_of_samples = -1
num_of_class = 20
anchors = [(128, 128), (90, 180), (180, 90), (256, 256), (180, 362), (362, 180), (512, 512), (362, 724), (724, 362)]

annotations = parse_annotations(ann_path, num_of_samples)
images, raw_images = get_images(img_path, annotations, keras.applications.vgg16.preprocess_input)
np.random.shuffle(images)

base_model = keras.applications.VGG16(weights='imagenet', include_top=False)
intermediate_layer = keras.layers.Conv2D(512, (3, 3), activation='relu')(base_model.output)
cls_output_len = 1*len(anchors)
reg_output_len = 4*len(anchors)
cls_layer = keras.layers.Conv2D(cls_output_len, (1, 1), activation='sigmoid')(intermediate_layer)
reg_layer = keras.layers.Conv2D(reg_output_len, (1, 1))(intermediate_layer)
model = keras.models.Model(inputs=base_model.input, outputs=keras.layers.concatenate([cls_layer, reg_layer]))

model.compile(optimizer='adam', loss=total_loss(cls_output_len, reg_output_len))

epoch = 5
for i in range(epoch):
    for image in images[:int(num_of_samples * 0.9)]:
        # receptive field of 3x3 area of last layer in VGG16 is 228
        y_pred = model.predict_on_batch(image[1])
        y_true = make_label(annotations[image[0]], y_pred, anchors)
        loss = model.train_on_batch(image[1], y_true)
        print('epoch:', i, ', loss', loss)
sess = tf.Session()
totalLoss = 0
for image in images[int(num_of_samples * 0.9):]:
    y_pred = model.predict_on_batch(image[1])
    y_true = make_label(annotations[image[0]], y_pred, anchors)
    totalLoss += model.evaluate(image[1], y_true)
    h = y_pred.shape[2]
    max_idx = np.argmax(np.apply_along_axis(lambda x: x[:len(anchors)].max(), 1, y_pred.reshape(-1, 5*len(anchors))))
    row = int(max_idx / h)
    col = max_idx - row * h
    target = y_pred[0][row][col]
    anchor_idx = np.argmax(target[:len(anchors)])
    x_a = 113 + 16 * col
    y_a = 113 + 16 * row
    bnd_box = target[anchor_idx:anchor_idx+4]
    bnd_box[0] = bnd_box[0] * anchors[anchor_idx][0] + x_a
    bnd_box[1] = bnd_box[1] * anchors[anchor_idx][1] + y_a
    bnd_box[2] = math.exp(bnd_box[2]) * anchors[anchor_idx][0]
    bnd_box[3] = math.exp(bnd_box[3]) * anchors[anchor_idx][1]
    raw_image = raw_images[image[0]]
    raw_image = keras.preprocessing.image.img_to_array(raw_image)
    x_1 = max((bnd_box[0] - bnd_box[2] / 2) / len(raw_image[0]), 0)
    y_1 = max((bnd_box[1] - bnd_box[3] / 2) / len(raw_image), 0)
    x_2 = min((bnd_box[0] + bnd_box[2] / 2) / len(raw_image[0]), 1)
    y_2 = min((bnd_box[1] + bnd_box[3] / 2) / len(raw_image), 1)
    image_with_bnd_box = tf.cast(tf.image.draw_bounding_boxes(np.expand_dims(raw_image, axis=0), np.array([[[x_1, y_1, x_2, y_2]]])), tf.uint8)
    image_encode = tf.image.encode_jpeg(image_with_bnd_box[0])
    file_name = tf.constant(image[0] + '.jpg')
    fwrite = tf.write_file(file_name, image_encode)
    sess.run(fwrite)
sess.close()
print('test loss:', totalLoss / (len(images) - int(num_of_samples * 0.9)))
