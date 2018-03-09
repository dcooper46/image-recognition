# -*- coding: utf-8 -*-
"""
    ppc-creative-genome
    keras
    retrain

    Retrain google's Inception V3 using Keras library
    
    @author: dancoope
    @created: 2/16/18
"""
from __future__ import division, print_function, absolute_import

import os
import argparse
import glob
import hashlib
import re
import warnings
import numpy as np

from sklearn.metrics import matthews_corrcoef
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.applications.inception_v3 import InceptionV3

from tensorflow.python.util import compat


# constants
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def split_train_validation_test(split_percentages, image_dir):
    validation_percentage, testing_percentage = split_percentages
    splits = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.abspath(
                    os.path.join(image_dir, dir_name, '*.' + extension)
            )
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, '
                  'which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. '
                  'Some images will never be selected.'.format(
                    dir_name, MAX_NUM_IMAGES_PER_CLASS)
            )
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = (hashlib
                                .sha1(compat.as_bytes(hash_name))
                                .hexdigest()
                                )
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)  # or basename
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        splits[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return splits


def load_images(image_dir, split_percentages):
    if len(split_percentages) != 2:
        warnings.warn("invalid testing splits: {}\nrunning with default of "
                      "[0.2, 0.1]".format(split_percentages))
        split_percentages = [0.2, 0.1]  # [validation, test]

    result = split_train_validation_test(split_percentages, image_dir)

    return result


def load_labels(labels_dir, labels_file):
    # optionally load all labels and store
    labels = open(labels_file).readlines()
    image_labels = {}
    for image_file in os.listdir(labels_dir):
        image_path = os.path.join(labels_dir, image_file)
        image_labels[image_file.strip('.txt')] = open(image_path).readlines()
    return image_labels, labels


def load_model(model_arch):
    if model_arch == 'inceptionV3':
        model = InceptionV3(include_top=False,
                            weights='imagenet',
                            pooling='avg')
        height, width = 299, 299
    else:
        raise NotImplementedError("That architecture is not implemented yet")
    return model, (height, width)


def get_top_model(base_model, n_labels):
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(n_labels, activation='sigmoid'))
    top_model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy']
    )
    return top_model


def save_bottleneck_features(model, dims, image_lists,
                             image_dir, bottleneck_dir):
    bottleneck_dir = os.path.abspath(bottleneck_dir)
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)

        for label_name, label_lists in image_lists.items():
            for eval_type in ['training', 'validation', 'testing']:
                print("creating {} bottlenecks".format(eval_type))
                images = label_lists[eval_type]
                for img in images:
                    img_file = os.path.join(image_dir, label_lists['dir'], img)
                    img_data = image.load_img(img_file, target_size=dims)
                    bottleneck_features = model.predict(
                            np.expand_dims(image.img_to_array(img_data), axis=0)
                    )
                    np.savetxt(os.path.join(bottleneck_dir, img + ".txt"),
                               bottleneck_features)


def encode_labels(image_labels, labels):
    return np.in1d(labels, image_labels).astype(int).tolist()


def read_bottleneck(img, dir_):
    return np.loadtxt(os.path.join(dir_, img + '.txt'))


def read_image_labels(labels_path):
    return open(labels_path).readlines()

#
# def get_image_labels_path(base_name, image_labels_dir):
#     full_path = os.path.join(base_name, image_labels_dir)
#     full_path += '.txt'
#     return full_path


def read_true_labels(image_labels_path, labels):
    image_labels = read_image_labels(image_labels_path)
    return encode_labels(image_labels, labels)


def get_true_labels(image_labels, labels):
    return encode_labels(image_labels, labels)


def get_xy_samples(images, labels, bottleneck_dir, image_labels_map):
    x = []
    y = []
    for i, img in enumerate(images):
        x.append(read_bottleneck(img, bottleneck_dir))
        y.append(get_true_labels(image_labels_map[img], labels))

    return np.array(x), np.array(y)


def save_model(model, filename):
    model_dir = os.path.dirname(filename)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(filename)


def main(args):
    for arg, val in vars(args).items():
        print("{a}:\t{v}".format(a=arg, v=val))

    split_percentages = [float(s.strip())
                         for s in args.split_percentages.split(",")]

    image_lists = load_images(args.image_dir, split_percentages)
    image_labels, labels = load_labels(args.labels_dir, args.labels_file)

    base_model, dims = load_model(args.model_arch)
    save_model(base_model, os.path.join(args.models_dir, "bottleneck_model.h5"))

    save_bottleneck_features(base_model, dims, image_lists,
                             args.image_dir, args.bottleneck_dir)

    model = get_top_model(base_model, len(labels))

    # get data
    X_train, y_train = get_xy_samples(image_lists['multi label']['training'],
                                      labels,
                                      args.bottleneck_dir,
                                      image_labels)

    X_valid, y_valid = get_xy_samples(image_lists['multi label']['validation'],
                                      labels,
                                      args.bottleneck_dir,
                                      image_labels)

    X_test, y_test = get_xy_samples(image_lists['multi label']['testing'],
                                    labels,
                                    args.bottleneck_dir,
                                    image_labels)

    print("Training: X ->{}\t y -> {}".format(X_train.shape, y_train.shape))
    print("Validation: X ->{}\t y -> {}".format(X_valid.shape, y_valid.shape))
    print("Testing: X ->{}\t y -> {}".format(X_test.shape, y_test.shape))
    # run model for transfer learning
    model.fit(X_train, y_train, batch_size=args.train_batch_size,
              epochs=args.epochs, shuffle=True,
              validation_data=(X_valid, y_valid))

    # outputs
    out_test = model.predict(X_test)

    # calc accuracies/fit - start basic
    threshold = 0.5
    rounded_outputs = np.array(
            [[1 if p >= threshold else 0 for p in prediction]
             for prediction in out_test]
    )
    correct = np.equal(rounded_outputs, y_test).sum(axis=1)  # type: np.ndarray
    accuracy = np.mean(correct / float(len(labels)))
    print("final accuracy on {} test images: {}".format(len(X_test), accuracy))

    thresholds = np.arange(0.2, 0.9, 0.1)
    accuracies, acc = [], []
    best_threshold = np.zeros(out_test.shape[1])
    for i in range(out_test.shape[1]):
        y_prob = out_test[:, i]
        for theta in thresholds:
            y_pred = [1 if prob >= theta else 0 for prob in y_prob]
            acc.append(matthews_corrcoef(y_test[:, i], y_pred))
        idx = np.argmax(acc)
        accuracies.append(np.max(acc))
        best_threshold[i] = thresholds[idx]
        acc = []

    y_pred = np.array([[1 if out_test[i, j] >= best_threshold[j] else 0 for j in
                        range(y_test.shape[1])] for i in range(len(y_test))])

    np.savetxt("predicted_labels_test", y_pred)
    save_model(model, os.path.join(args.models_dir, "prediction_model.h5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--labels_dir", type=str)
    parser.add_argument("--labels_file", type=str)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--bottleneck_dir", type=str, default="bottlenecks")
    parser.add_argument("--model_arch", type=str, default="inceptionV3")
    parser.add_argument("--split_percentages", type=str, default="20, 10",
                        help="validation and testing percentages as value "
                             "between 0-100")

    pargs = parser.parse_args()

    main(pargs)
