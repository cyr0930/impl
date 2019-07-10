import os
import warnings
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.xception import Xception, preprocess_input
from tensorflow.python.keras import layers, models, optimizers, callbacks, backend, utils
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

print(os.listdir("../input"))

DATA_PATH = '../../data/2019-3rd-ml-month-with-kakr'
IMG_PATH = '../../data/2019-3rd-ml-month-with-kakr'
OUT_PATH = '../../data/2019-3rd-ml-month-with-kakr'
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')
TRAIN_CROPPED_PATH = os.path.join(DATA_PATH, 'train_cropped')
TEST_CROPPED_PATH = os.path.join(DATA_PATH, 'test_cropped')

nrows = None
img_size = (299, 299)
batch_size = 32
epochs = 100
train_ratio = 0.8

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), nrows=nrows)
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))


def get_steps(num_samples, batch_size):
    return num_samples // batch_size + int(num_samples % batch_size > 0)


def crop_boxing_img(data, path, path_cropped, margin=16, size=img_size):
    for i, row in data.iterrows():
        img_name = row['img_file']
        img = Image.open(os.path.join(path, img_name))
        pos = data.loc[data["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)
        width, height = img.size
        x1 = max(0, pos[0] - margin)
        y1 = max(0, pos[1] - margin)
        x2 = min(pos[2] + margin, width)
        y2 = min(pos[3] + margin, height)
        cropped = img.crop((x1, y1, x2, y2)).resize(size)
        cropped.save(os.path.join(path_cropped, img_name))


def lr_cos(step, interval, min_lr, max_lr):
    return (math.cos(step * math.pi / (interval - 1)) + 1) * (max_lr - min_lr) / 2 + min_lr


# crop and save
if not os.path.isdir(TRAIN_CROPPED_PATH):
    os.makedirs(TRAIN_CROPPED_PATH)
    crop_boxing_img(df_train, TRAIN_IMG_PATH, TRAIN_CROPPED_PATH)
    os.makedirs(TEST_CROPPED_PATH)
    crop_boxing_img(df_test, TEST_IMG_PATH, TEST_CROPPED_PATH)

df_train["class"] = df_train["class"].astype('str')
df_train = df_train[['img_file', 'class']]
df_test = df_test[['img_file']]

train_datagen = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=False, zoom_range=0.1, rotation_range=20, fill_mode='nearest',
    width_shift_range=0.2, height_shift_range=0.2,
    preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

weights, probs = 0, 0
num_models = 2
its = np.arange(df_train.shape[0])
classes = list(sorted([str(i+1) for i in range(df_class.shape[0])]))
kf = StratifiedKFold(n_splits=5, shuffle=True)
final_metrics = []
for train_idx, val_idx in kf.split(its, df_train['class']):
    print('fold' + str(num_models))
    if num_models == 0:
        break
    num_models -= 1

    X_train = df_train.iloc[train_idx, :].reset_index(drop=True)
    X_val = df_train.iloc[val_idx, :].reset_index(drop=True)

    nb_train_samples = len(X_train)
    nb_validation_samples = len(X_val)
    nb_test_samples = len(df_test)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=X_train, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
        classes=classes, class_mode='categorical', batch_size=batch_size
    )
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=X_val, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
        classes=classes, class_mode='categorical', batch_size=batch_size, shuffle=False
    )
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test, directory=TEST_CROPPED_PATH, x_col='img_file', y_col=None, target_size=img_size,
        class_mode=None, batch_size=batch_size, shuffle=False
    )

    base_model = Xception(include_top=False, pooling='avg')
    output = layers.Dense(512, activation='relu')(base_model.output)
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(len(df_class), activation='softmax')(output)
    model = models.Model(inputs=base_model.input, outputs=output)

    lr = 0.0001
    model.compile(optimizer=optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['acc'])
    es = callbacks.EarlyStopping(patience=3, mode='min', verbose=1)
    rp = callbacks.ReduceLROnPlateau(factor=0.5, patience=1, min_lr=lr / 10, mode='min', verbose=1)
    history1 = model.fit_generator(
        train_generator, steps_per_epoch=get_steps(nb_train_samples, batch_size),
        validation_data=validation_generator, validation_steps=get_steps(nb_validation_samples, batch_size),
        epochs=epochs, verbose=1, callbacks=[es, rp],
    )

    lr = lr * 10
    model.compile(optimizer=optimizers.SGD(lr, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
    es = callbacks.EarlyStopping(patience=3, mode='min', verbose=1)
    rp = callbacks.ReduceLROnPlateau(factor=0.5, patience=1, min_lr=lr / 10, mode='min', verbose=1)
    history2 = model.fit_generator(
        train_generator, steps_per_epoch=get_steps(nb_train_samples, batch_size),
        validation_data=validation_generator, validation_steps=get_steps(nb_validation_samples, batch_size),
        epochs=epochs, verbose=1, callbacks=[es, rp]
    )

    w = 0.5 ** len(history2.history['val_acc'])
    for acc in history2.history['val_acc']:
        w *= 2
        weights += acc * w
        probs += acc * w * model.predict_generator(generator=test_generator,
                                                   steps=get_steps(nb_test_samples, batch_size), verbose=1)
    final_metrics.append((history2.history['loss'][-1], history2.history['acc'][-1],
                          history2.history['val_loss'][-1], history2.history['val_acc'][-1]))

for i in range(len(final_metrics)):
    t = final_metrics[i]
    print('Fold ' + str(i))
    print('Loss: %.4f, Acc: %.4f for Training' % (t[0], t[1]))
    print('Loss: %.4f, Acc: %.4f for Validation' % (t[2], t[3]))

csv_probs = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
probs_mean = probs / weights
for i in range(len(df_class)):
    csv_probs["probs" + str(i+1)] = probs_mean[:, i]
csv_probs.to_csv("z_probs.csv", index=False)

preds = np.argmax(probs_mean, axis=1)
preds = np.array([classes[i] for i in preds])
submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
submission["class"] = preds
submission.to_csv(os.path.join(OUT_PATH, "submission.csv"), index=False)
print('done')

# from IPython.display import HTML
# import base64
# def create_download_link(df, title="Download CSV file", filename="submission.csv"):
#     csv = df.to_csv()
#     b64 = base64.b64encode(csv.encode())
#     payload = b64.decode()
#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
#     html = html.format(payload=payload, title=title, filename=filename)
#     return HTML(html)
# create_download_link(submission)
