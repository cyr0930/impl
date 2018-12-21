import tensorflow as tf
from tensorflow import keras
from object_detection.faster_rcnn.utils import parse_annotations, get_images, make_label
from object_detection.faster_rcnn.losses import total_loss

root_dir = '../../data/VOC2007'
ann_path = root_dir + '/Annotations'
img_path = root_dir + '/JPEGImages'
num_of_samples = 16
num_of_class = 20
anchors = [(128, 128), (90, 180), (180, 90), (256, 256), (180, 362), (362, 180), (512, 512), (362, 724), (724, 362)]

annotations = parse_annotations(ann_path, num_of_samples)
images = get_images(img_path, annotations, keras.applications.vgg16.preprocess_input)

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
totalLoss = 0
for image in images[int(num_of_samples * 0.9):]:
    y_pred = model.predict_on_batch(image[1])
    y_true = make_label(annotations[image[0]], y_pred, anchors)
    totalLoss += model.evaluate(image[1], y_true)
    # bnd_boxes = y_pred[:, :, :, len(anchors):]
    # image_with_bnd_box = tf.image.draw_bounding_boxes(image, bnd_boxes)
print('test loss:', totalLoss / 10)
