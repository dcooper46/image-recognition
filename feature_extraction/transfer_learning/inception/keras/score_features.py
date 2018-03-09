# -*- coding: utf-8 -*-
"""
    ppc-creative-genome
    keras
    score_features

    given a new set of images/assets, extract bottlenecks
    and predict probabilities for each labeled attribute
    
    @author: dancoope
    @created: 2/20/18
"""
from __future__ import division, print_function, absolute_import

import sys
import os
import ast
import argparse
import pandas as pd
import numpy as np

from utils import safe_int

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image


def read_labels(labels_file):
    # should be the same labels file the prediction model was trained on
    label_lines = [line.rstrip() for line in np.loadtxt(labels_file, dtype=str)]
    return label_lines


def get_image_data(img, dims):
    image_data = image.img_to_array(image.load_img(img, target_size=dims))
    return np.expand_dims(image_data, axis=0)


def load_images(image_dir, dims):
    _, _, images = os.walk(image_dir).next()
    images = filter(lambda x: not x.startswith('.'), images)
    images = [os.path.join(image_dir, img) for img in images]
    image_data = [get_image_data(img, dims) for img in images]
    return zip(images, image_data)


def load_saved_model(model_dir, model_name):
    model_dir = os.path.abspath(model_dir)
    if not model_name.endswith(".h5"):
        model_name += ".h5"

    model_path = os.path.join(model_dir, model_name)
    return load_model(model_path)


def create_bottleneck(model, img):
    return model.predict(img)


def save_bottleneck(bottleneck, filename):
    np.savetxt(filename, bottleneck)


def score_image(model, features):
    predictions = model.predict(features).reshape((-1))
    sorted_ids = predictions.argsort()[-len(predictions):][::-1]
    return zip(sorted_ids, predictions[sorted_ids])


def save_scores(img_file, scores):
    # scores is list/array of tuples: (feature description, score)
    with open(img_file, 'wb') as f:
        for feat, score in scores:
            f.write("{feat}\t{score}\n".format(feat=feat, score=score))


def output_scores(scored_images, scores_dir):
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)
    scored_images_df = pd.DataFrame.from_dict(scored_images)
    columns = scored_images_df.columns.tolist()
    columns.sort(key=lambda x: (''.join(i for i in x if not i.isdigit()),
                                safe_int(''.join(i for i in x if i.isdigit()))
                                )
                 )
    scored_images_df.reindex_axis(columns, axis=1)
    scored_images_df.to_csv(os.path.join(scores_dir, 'image_scores.csv'))


def main(args):
    for arg, val in vars(args).items():
        print("{a}:\t{v}".format(a=arg, v=val))

    images = load_images(args.image_dir, args.image_dims)
    n_images = len(images)
    print("{} images loaded".format(n_images))

    labels = read_labels(args.labels_file)
    print("{} labels loaded".format(len(labels)))

    bottleneck_model = load_saved_model(args.models_dir,
                                        args.bottleneck_model_name)
    prediction_model = load_saved_model(args.models_dir,
                                        args.prediction_model_name)

    scored_images = {}
    print("scoring images...")
    for i, it in enumerate(images):
        img, img_data = it
        basename = os.path.splitext(os.path.basename(img))[0]

        bottleneck = create_bottleneck(bottleneck_model, img_data)
        bottleneck_file = os.path.join(args.bottleneck_dir, basename + ".txt")
        save_bottleneck(bottleneck, bottleneck_file)

        scores = score_image(prediction_model, bottleneck)
        scores = [(labels[idx], score) for idx, score in scores]
        scores_file = os.path.join(args.scores_dir, basename + ".txt")
        save_scores(scores_file, scores)
        scored_images[basename] = dict(scores)

        if i % 10 == 0:
            print("\tprocessed {}/{} images".format(i, n_images))

    output_scores(scored_images, args.scores_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--labels_file', type=str)
    parser.add_argument('--models_dir', type=str)
    parser.add_argument('--bottleneck_model_name',
                        type=str, default='bottleneck_model')
    parser.add_argument('--prediction_model_name',
                        type=str, default='prediction_model')
    parser.add_argument('--bottleneck_dir', type=str, default='bottlenecks')
    parser.add_argument('--scores_dir', type=str, default='feature_scores')
    parser.add_argument('--image_dims', type=ast.literal_eval,
                        default=(299, 299))

    pargs = parser.parse_args()

    main(pargs)
    sys.exit()
