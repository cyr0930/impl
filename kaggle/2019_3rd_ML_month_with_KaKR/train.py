import os
import warnings
import math
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import xception, densenet, inception_v3
from tensorflow.python.keras import layers, models, optimizers, callbacks, utils
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

DATA_PATH = '../../data/2019-3rd-ml-month-with-kakr'
IMG_PATH = '../../data/2019-3rd-ml-month-with-kakr'
OUT_PATH = '../../data/2019-3rd-ml-month-with-kakr'
MODEL_PATH = OUT_PATH
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')
TRAIN_CROPPED_PATH = os.path.join(IMG_PATH, 'train_cropped')

model_name = 'xception'
pretrained_model = None
img_size = None
preprocess_input = densenet.preprocess_input
if model_name == 'densenet':
    pretrained_model = densenet.DenseNet201
    img_size = (224, 224)
    preprocess_input = densenet.preprocess_input
elif model_name == 'xception':
    pretrained_model = xception.Xception
    img_size = (299, 299)
    preprocess_input = xception.preprocess_input
elif model_name == 'inception':
    pretrained_model = inception_v3.InceptionV3
    img_size = (299, 299)
    preprocess_input = inception_v3.preprocess_input
nrows = None
batch_size = 32
epochs = 100

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), nrows=nrows)
df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))


def get_steps(num_samples, batch_size):
    return num_samples // batch_size + int(num_samples % batch_size > 0)


def crop_boxing_img(data, path, path_cropped, margin=0):
    for i, row in data.iterrows():
        img_name = row['img_file']
        img = Image.open(os.path.join(path, img_name))
        pos = data.loc[data["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)
        width, height = img.size
        x1 = max(0, pos[0] - margin)
        y1 = max(0, pos[1] - margin)
        x2 = min(pos[2] + margin, width)
        y2 = min(pos[3] + margin, height)
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(os.path.join(path_cropped, img_name))


def lr_cos(step, interval, min_lr, max_lr):
    if interval == 1:
        return max_lr
    return (math.cos(step * math.pi / (interval - 1)) + 1) * (max_lr - min_lr) / 2 + min_lr


def make_dataset(df, path):
    x, y = [], None
    for i, image_path in enumerate(df['img_file'].values):
        image_path = path + '/' + image_path
        x.append(Image.open(image_path).convert("RGB"))
    if 'class' in df:
        y = utils.to_categorical(df['class'].astype(int)-1, num_classes=len(df_class))
    return x, y


def create_model(pretrained, img_size):
    base_model = pretrained(input_shape=(*img_size, 3), include_top=False, pooling='avg')
    output = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(base_model.output)
    output = layers.Dropout(0.2)(output)
    output = layers.Dense(len(df_class), activation='softmax', kernel_initializer='he_normal')(output)
    return models.Model(inputs=base_model.input, outputs=output)


if not os.path.isdir(TRAIN_CROPPED_PATH):
    os.makedirs(TRAIN_CROPPED_PATH)
    crop_boxing_img(df_train, TRAIN_IMG_PATH, TRAIN_CROPPED_PATH)

df_train["class"] = df_train["class"].astype('str')
df_train = df_train[['img_file', 'class']]

train_datagen = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=False, zoom_range=0.1, rotation_range=20,
    width_shift_range=0.2, height_shift_range=0.2,
    preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

num_models, count = 5, 0
its = np.arange(df_train.shape[0])
classes = list(sorted([str(i + 1) for i in range(df_class.shape[0])]))
kf = StratifiedKFold(n_splits=5, shuffle=True)
final_metrics = []
for train_idx, val_idx in kf.split(its, df_train['class']):
    print('Fold: %d, num_train: %d, num_val: %d' % (count, len(train_idx), len(val_idx)))
    if count == num_models:
        break

    X_train = df_train.iloc[train_idx, :].reset_index(drop=True)
    X_val = df_train.iloc[val_idx, :].reset_index(drop=True)

    nb_train_samples = len(X_train)
    nb_validation_samples = len(X_val)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=X_train, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
        classes=classes, class_mode='categorical', batch_size=batch_size
    )
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=X_val, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
        classes=classes, class_mode='categorical', batch_size=batch_size, shuffle=False
    )

    model = create_model(pretrained_model, img_size)

    lr = 0.0002
    model.compile(optimizer=optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['acc'])
    es = callbacks.EarlyStopping(patience=0, mode='min', verbose=1)
    model.fit_generator(
        train_generator, steps_per_epoch=get_steps(nb_train_samples, batch_size),
        validation_data=validation_generator, validation_steps=get_steps(nb_validation_samples, batch_size),
        epochs=epochs, verbose=2, callbacks=[es],
    )

    lr = lr * 10
    model.compile(optimizer=optimizers.SGD(lr, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
    es = callbacks.EarlyStopping(patience=1, mode='min', verbose=1)
    rp = callbacks.ReduceLROnPlateau(factor=0.5, patience=0, min_lr=lr / 1000, mode='min', verbose=1)
    ckpt = callbacks.ModelCheckpoint(os.path.join(MODEL_PATH, '%s_%d_{epoch}_{val_acc}.ckpt' % (model_name, count)),
                                     save_weights_only=True)
    history = model.fit_generator(
        train_generator, steps_per_epoch=get_steps(nb_train_samples, batch_size),
        validation_data=validation_generator, validation_steps=get_steps(nb_validation_samples, batch_size),
        epochs=epochs, verbose=2, callbacks=[es, rp, ckpt]
    )

    count += 1
    final_metrics.append((history.history['loss'][-1], history.history['acc'][-1],
                          history.history['val_loss'][-1], history.history['val_acc'][-1]))

for i in range(len(final_metrics)):
    t = final_metrics[i]
    print('Fold ' + str(i))
    print('Loss: %.4f, Acc: %.4f for Training' % (t[0], t[1]))
    print('Loss: %.4f, Acc: %.4f for Validation' % (t[2], t[3]))

zip_file_name = os.path.join(OUT_PATH, model_name + '.zip')
with ZipFile(zip_file_name, 'w') as zip_file:
    for file_name in os.listdir(OUT_PATH):
        abs_path = os.path.join(OUT_PATH, file_name)
        if os.path.isfile(abs_path) and abs_path != zip_file_name:
            zip_file.write(abs_path)
            os.remove(abs_path)
