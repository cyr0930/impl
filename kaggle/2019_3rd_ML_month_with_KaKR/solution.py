import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.python.keras import layers, models, optimizers, backend, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

DATA_PATH = '../../data/2019-3rd-ml-month-with-kakr/'
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')
TRAIN_CROPPED_PATH = os.path.join(DATA_PATH, 'train_cropped')
TEST_CROPPED_PATH = os.path.join(DATA_PATH, 'test_cropped')

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))

img_size = (299, 299)
batch_size = 20
epochs = 1
train_ratio = 0.8
rand_seed = 42


# img_size에 맞추다 보니 가로세로 비율이 이상해지는데 이게 좋은가 여백을 두는게 좋은가?
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


# crop and save
if not os.path.isdir(TRAIN_CROPPED_PATH):
    os.makedirs(TRAIN_CROPPED_PATH)
    crop_boxing_img(df_train, TRAIN_IMG_PATH, TRAIN_CROPPED_PATH)
    os.makedirs(TEST_CROPPED_PATH)
    crop_boxing_img(df_test, TEST_IMG_PATH, TEST_CROPPED_PATH)


df_train["class"] = df_train["class"].astype('str')
df_train = df_train[['img_file', 'class']]
df_test = df_test[['img_file']]

its = np.arange(df_train.shape[0])
train_idx, val_idx = train_test_split(its, train_size=train_ratio, random_state=rand_seed)

X_train = df_train.iloc[train_idx, :].reset_index(drop=True)
X_val = df_train.iloc[val_idx, :].reset_index(drop=True)

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
nb_test_samples = len(df_test)

train_datagen = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=False, zoom_range=0.1, rotation_range=20, fill_mode='nearest',
    width_shift_range=0.2, height_shift_range=0.2, preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=X_train, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
    color_mode='rgb', class_mode='categorical', batch_size=batch_size, seed=rand_seed
)
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=X_val, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
    color_mode='rgb', class_mode='categorical', batch_size=batch_size, shuffle=False
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test, directory=TEST_CROPPED_PATH, x_col='img_file', y_col=None, target_size=img_size,
    color_mode='rgb', class_mode=None, batch_size=batch_size, shuffle=False
)


def get_steps(num_samples, batch_size):
    return num_samples // batch_size + int(num_samples % batch_size > 0)


probs = []
num_models = 2
for i in range(num_models):
    base_model = MobileNetV2(include_top=False)
    output = layers.concatenate([
        layers.GlobalAveragePooling2D()(base_model.output),
        layers.GlobalMaxPooling2D()(base_model.output)
    ])
    output = layers.Dense(512, activation='relu')(output)
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(196, activation='softmax')(output)
    model = models.Model(inputs=base_model.input, outputs=output)

    # filepath = "my_resnet_model_{val_acc:.2f}_{val_loss:.4f}.h5"
    # ckpt = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
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

    test_generator.reset()
    probs.append(model.predict_generator(generator=test_generator,
                                         steps=get_steps(nb_test_samples, batch_size), verbose=1))

    its = np.arange(X_train.shape[0])
    buf = val_idx
    train_idx, val_idx = train_test_split(its, train_size=X_train.shape[0]-X_val.shape[0], random_state=rand_seed)
    train_idx = np.concatenate([train_idx, buf])
    X_train = df_train.iloc[train_idx, :].reset_index(drop=True)
    X_val = df_train.iloc[val_idx, :].reset_index(drop=True)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=X_train, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
        color_mode='rgb', class_mode='categorical', batch_size=batch_size, seed=rand_seed
    )
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=X_val, directory=TRAIN_CROPPED_PATH, x_col='img_file', y_col='class', target_size=img_size,
        color_mode='rgb', class_mode='categorical', batch_size=batch_size, shuffle=False
    )

csv_probs = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
probs_sum = 0
for prob in probs:
    probs_sum += prob
probs_mean = probs_sum / len(probs)
for i in range(len(df_class)):
    csv_probs["probs" + str(i+1)] = probs_mean[:, i]
csv_probs.to_csv("probs.csv", index=False)

indice = np.argmax(probs_mean, axis=1)
labels = dict((v, k) for k, v in train_generator.class_indices.items())
preds = [labels[k] for k in indice]
submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
submission["class"] = preds
submission.to_csv(os.path.join(DATA_PATH, "submission.csv"), index=False)
print('done')
