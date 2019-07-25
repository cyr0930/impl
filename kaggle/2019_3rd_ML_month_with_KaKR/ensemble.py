import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import xception, densenet
from tensorflow.python.keras import models

INPUT_PATH = '../../data'
DATA_PATH = os.path.join(INPUT_PATH, '2019-3rd-ml-month-with-kakr')
OUT_PATH = os.path.join(INPUT_PATH, '2019-3rd-ml-month-with-kakr')
MODEL_PATH = os.path.join(INPUT_PATH, '2019-3rd-ml-month-with-kakr')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')
TEST_CROPPED_PATH = os.path.join(OUT_PATH, 'test_cropped')

batch_size = 32
pretrained_params = [
    ('xception', (299, 299), xception.preprocess_input),
    ('densenet', (224, 224), densenet.preprocess_input)
]
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
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


if not os.path.isdir(TEST_CROPPED_PATH):
    os.makedirs(TEST_CROPPED_PATH)
    crop_boxing_img(df_test, TEST_IMG_PATH, TEST_CROPPED_PATH)

df_test = df_test[['img_file']]
classes = list(sorted([str(i + 1) for i in range(df_class.shape[0])]))

weights, probs = 0, 0
for param in pretrained_params:
    model_name, img_size, preprocess_input = param
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen_flip = ImageDataGenerator(preprocessing_function=lambda x: preprocess_input(np.fliplr(x)))
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test, directory=TEST_CROPPED_PATH, x_col='img_file', y_col=None, target_size=img_size,
        class_mode=None, batch_size=batch_size, shuffle=False
    )
    test_generator_flip = test_datagen_flip.flow_from_dataframe(
        dataframe=df_test, directory=TEST_CROPPED_PATH, x_col='img_file', y_col=None, target_size=img_size,
        class_mode=None, batch_size=batch_size, shuffle=False
    )
    for fold in range(5):
        cur_list = []
        path = os.path.join(MODEL_PATH, 'car-' + model_name)
        for file_name in os.listdir(path):
            if file_name.startswith(model_name + '_' + str(fold)):
                cur_list.append(file_name)
        cur_list.sort()
        w = 0.5 ** len(cur_list)
        for file_name in cur_list:
            test_generator.reset()
            test_generator_flip.reset()
            acc = float('0.' + file_name.split('_')[3].split('.')[1])
            w *= 2
            weights += acc * w * 2
            steps = get_steps(len(df_test), batch_size)
            model = models.load_model(os.path.join(path, file_name))
            probs += acc * w * model.predict_generator(generator=test_generator, steps=steps, verbose=1)
            probs += acc * w * model.predict_generator(generator=test_generator_flip, steps=steps, verbose=1)

probs_mean = probs / weights
preds = np.argmax(probs_mean, axis=1)
preds = np.array([classes[i] for i in preds])
submission = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))
submission["class"] = preds
submission.to_csv(os.path.join(OUT_PATH, "submission.csv"), index=False)
print('done')
