# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple transfer learning with an Inception v3 architecture model which
displays summaries in TensorBoard.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:

bazel build third_party/tensorflow/examples/image_retraining:retrain && \
bazel-bin/third_party/tensorflow/examples/image_retraining/retrain \
--image_dir ~/flower_photos

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.


To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

call:
    python retrain.py --bottleneck_dir=bottlenecks
    --how_many_training_steps 1500 --model_dir=model_dir
    --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt
    --summaries_dir=retrain_logs --image_dir=images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob
import hashlib
import random
import re
import sys
import tarfile

from datetime import datetime
from six.moves import urllib
from tensorflow.python.util import compat
from tensorflow.python.framework import graph_util

from utils import *


# Directory containing files with correct image labels for each image.
IMAGE_LABELS_DIR = "/Users/dancoope/Documents/image_recognition" \
                   "/image_labels_dir"
# Contains list of all labels where each label is on a separate line
ALL_LABELS_FILE = "/Users/dancoope/Documents/image_recognition/labels.txt"


def main(_):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # Set up the pre-trained graph.
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph())

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(os.path.abspath(FLAGS.image_dir),
                                     FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)
    for eval_type, imgs in image_lists.items():
        print("{}: {}".format(eval_type, len(imgs)))

    if len(image_lists.keys()) == 0:
        print(
            'Folder containing training images has not been found inside '
            '{} directory. \n'
            'Put all the training images into one folder inside {} directory '
            'and delete everything else inside the {} directory.'
            .format(FLAGS.image_dir, FLAGS.image_dir, FLAGS.image_dir))
        return -1

    if len(image_lists.keys()) > 1:
        print(
            'More than one folder found inside {} directory. \n'
            'In order to prevent validation issues, put all the training images'
            ' into one folder inside {} directory and delete everything else '
            'inside the {} directory.'
            .format(FLAGS.image_dir, FLAGS.image_dir, FLAGS.image_dir))
        return -1

    if not os.path.isfile(ALL_LABELS_FILE):
        print(
            'File {} containing all possible labels (classes) does not exist.\n'
            'Create it in project root and put each possible label on new line,'
            ' it is exactly the same as creating an image_label file for image '
            'that is in all the possible classes.'.format(ALL_LABELS_FILE))
        return -1

    with open(ALL_LABELS_FILE) as f:
        labels = f.read().splitlines()
    class_count = len(labels)

    if class_count == 0:
        print(
            'No valid labels inside file {} that should contain all possible '
            'labels (= classes).'.format(ALL_LABELS_FILE))
        return -1
    if class_count == 1:
        print(
            'Only one valid label found inside {} - multiple classes are '
            'needed for classification.'.format(ALL_LABELS_FILE))
        return -1

    do_distort_images = should_distort_images(
            FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
            FLAGS.random_brightness)

    sess = tf.Session()

    if do_distort_images:
        distorted_jpeg_data_tensor, distorted_image_tensor = (
            add_input_distortions(FLAGS.flip_left_right,
                                  FLAGS.random_crop,
                                  FLAGS.random_scale,
                                  FLAGS.random_brightness)
        )
    else:
        cache_bottlenecks(sess, image_lists,
                          FLAGS.image_dir, FLAGS.bottleneck_dir,
                          jpeg_data_tensor, bottleneck_tensor)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(class_count,
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to specified location
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation'
    )

    # Set up all our weights to their initial default values.
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(FLAGS.how_many_training_steps):

        if do_distort_images:
            train_bottlenecks, train_ground_truth = (
                get_random_distorted_bottlenecks(
                        sess, image_lists, FLAGS.train_batch_size, 'training',
                        FLAGS.image_dir, distorted_jpeg_data_tensor,
                        distorted_image_tensor, resized_image_tensor,
                        bottleneck_tensor)
            )
        else:
            train_bottlenecks, train_ground_truth = (
                get_random_cached_bottlenecks(
                        sess, image_lists, FLAGS.train_batch_size, 'training',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        bottleneck_tensor, labels)
            )

        train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth}
        )
        train_writer.add_summary(train_summary, i)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth}
            )
            print('%s: Step %d: Train accuracy = %.1f%%' % (
                datetime.now(), i, train_accuracy * 100
            )
                  )
            print('%s: Step %d: Cross entropy = %f' % (
                datetime.now(), i, cross_entropy_value
            )
                  )
            validation_bottlenecks, validation_ground_truth = (
                get_random_cached_bottlenecks(
                        sess, image_lists,
                        FLAGS.validation_batch_size, 'validation',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        bottleneck_tensor, labels
                )
            )

            validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth}
            )
            validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%%' % (
                datetime.now(), i, validation_accuracy * 100
            )
                  )

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    print("Test batch size: {}".format(FLAGS.test_batch_size))
    test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            bottleneck_tensor, labels)
    test_accuracy = sess.run(
            evaluation_step,
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

    # Write out the trained graph and labels with the weights stored
    output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')


def maybe_download_and_extract():
    """
  Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    print("inside maybe_download_and_extract")
    dest_directory = FLAGS.model_dir
    print("dest_directory: {}".format(dest_directory))
    if not os.path.exists(dest_directory):
        print("making model directory")
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        print("need to download")

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename,
                float(count * block_size) / float(total_size) * 100.0
            )
                             )
        sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    print("necessary files downloaded")
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_inception_graph():
    """
    Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
    """
    with tf.Session() as sess:
        model_filename = os.path.join(FLAGS.model_dir,
                                      'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
            tf.import_graph_def(graph_def, name='', return_elements=[
                BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                RESIZED_INPUT_TENSOR_NAME])
        )
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

    Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
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
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
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
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put an image in, the data set creator has a way of
    # grouping photos that are close variations of each other. For example
    # this is used in the plant disease data set to group multiple pictures of
    # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
            hash_name_hashed = (hashlib
                                .sha1(compat.as_bytes(hash_name))
                                .hexdigest()
                                )
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    for label_name in result.keys():
        print(label_name)
        print("train: {}".format(len(result[label_name]['training'])))
        print("validation: {}".format(len(result[label_name]['validation'])))
        print("testing: {}".format(len(result[label_name]['testing'])))
    return result


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """
  Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
    """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """
  Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.

  Returns:
    The jpeg input layer and the distorted result tensor.
    """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [MODEL_INPUT_HEIGHT,
                                    MODEL_INPUT_WIDTH,
                                    MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
    """
    Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    bottleneck_tensor: The penultimate output layer of the graph.

    Returns:
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                bottneck = get_or_create_bottleneck(
                        sess, image_lists, label_name, index,
                        image_dir, category, bottleneck_dir,
                        jpeg_data_tensor, bottleneck_tensor)
                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    print(str(how_many_bottlenecks) +
                          ' bottleneck files created.')


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    """
  Adds a new sigmoid and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
                name='BottleneckInputPlaceholder'
        )

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            layer_weights = tf.Variable(tf.truncated_normal(
                    [BOTTLENECK_TENSOR_SIZE, class_count],
                    stddev=0.001
            ),
                    name='final_weights')
            variable_summaries(layer_weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]),
                                       name='final_biases')
            variable_summaries(layer_biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram(layer_name + '/pre_activations', logits)

    final_tensor = tf.nn.sigmoid(logits, name=final_tensor_name)
    tf.summary.histogram(final_tensor_name + '/activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=ground_truth_input)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        train_step = (tf.train
                      .GradientDescentOptimizer(FLAGS.learning_rate)
                      .minimize(cross_entropy_mean)
                      )

    return (train_step, cross_entropy_mean, bottleneck_input,
            ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
  Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Nothing.
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # tf.argmax(result_tensor, 1) = return index of maximal value
            # (= 1 in a 1-of-N encoding vector) in each row (axis = 1)
            # But we have more ones (indicating multiple labels) in one row
            # of result_tensor due to the multi-label classification
            # correct_prediction = tf.equal(tf.argmax(result_tensor, 1), \
            #   tf.argmax(ground_truth_tensor, 1))

            # ground_truth is not a binary tensor, it contains the probabilities
            #  of each label = we need to tf.round() it
            # to acquire a binary tensor allowing comparison by tf.equal()
            # See: http://stackoverflow.com/questions/39219414/in-tensorflow-how-can-i-get-nonzero-values-and-their-indices-from-a-tensor-with
            # TODO: adjust to take different threshold (other than 0.5)
            correct_prediction = tf.equal(tf.round(result_tensor),
                                          ground_truth_tensor)
        with tf.name_scope('accuracy'):
            # Mean accuracy over all labels:
            # http://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation
            evaluation_step = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32)
            )
        tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step


def get_random_distorted_bottlenecks(sess, image_lists, how_many, category,
                                     image_dir, input_jpeg_tensor,
                                     distorted_image, resized_input_tensor,
                                     bottleneck_tensor, labels):
    """
    Retrieves bottleneck values for training images, after distortions.

    If we're training with distortions like crops, scales, or flips, we have to
    recalculate the full model for every image, and so we can't use cached
    bottleneck values. Instead we find random images for the requested category,
    run them through the distortion graph, and then the full graph to get the
    bottleneck results for each.

    Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
    labels: All possible labels loaded from file labels.txt.

    Returns:
    List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(labels)
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = 0  # there is only one folder with images = 'multi-label'
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists,
                                    label_name,
                                    image_index,
                                    image_dir,
                                    category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()

        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                             resized_input_tensor,
                                             bottleneck_tensor)

        labels_file = get_image_labels_path(image_lists,
                                            label_name,
                                            image_index,
                                            IMAGE_LABELS_DIR,
                                            category)
        ground_truth = get_ground_truth(labels_file, labels, class_count)

        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor, labels):
    """
    Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The number of bottleneck values to return.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
    labels: All possible labels loaded from file labels.txt.

    Returns:
    List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(labels)
    bottlenecks = []
    ground_truths = []
    if how_many >= 0:
        for unused_i in range(how_many):
            label_index = 0  # only one folder with images = 'multi-label'
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name,
                    image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor,
                    bottleneck_tensor
            )
            labels_file = get_image_labels_path(image_lists,
                                                label_name,
                                                image_index,
                                                IMAGE_LABELS_DIR,
                                                category)
            ground_truth = get_ground_truth(labels_file, labels, class_count)

            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    else:
        # retrieve all bottlenecks
        for unused_i, _ in enumerate(image_lists.keys()):
            label_index = 0  # only one folder with images = 'multi-label'
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name,
                    image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor,
                    bottleneck_tensor
            )
            labels_file = get_image_labels_path(image_lists,
                                                label_name,
                                                image_index,
                                                IMAGE_LABELS_DIR,
                                                category)
            ground_truth = get_ground_truth(labels_file, labels, class_count)

            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
    )
    parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
    )
    parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
    )
    parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
    )
    parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
    )
    parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
      '--validation_percentage',
      type=int,
      default=20,
      help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
    )
    parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
    )
    parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
    )
    parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
    )
    parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    parser.add_argument(
      '--architecture',
      type=str,
      default='inception_v3',
      help="""\
Which model architecture to use. 'inception_v3' is the most accurate, but
also the slowest. For faster or smaller models, chose a MobileNet with the
form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
less accurate, but smaller and faster network that's 920 KB on disk and
takes 128x128 images. See:
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
for more information on Mobilenet.\
      """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
