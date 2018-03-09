"""
call:
    python score_features.py <location to images> <location to labels>

for now, assume this is being run from the same directory
that the trained model files are in. Output scores to current directory so
its obvious which model created them. Only take locations to images and labels
for now.  Will formalize later.
"""

import tensorflow as tf
import sys
import os
import pandas as pd

from utils import safe_int


image_dir = os.path.abspath(sys.argv[1])
labels_file = os.path.abspath(sys.argv[2])
scores_dir = os.path.abspath('feature_scores')

_, _, images = os.walk(image_dir).next()
images = filter(lambda x: not x.startswith('.'), images)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile(labels_file)]


# Unpersist graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

scored_images = {}
with tf.Session() as sess:
    result_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    for image in images:
        print("getting feature scores for {}".format(image))
        image_path = os.path.join(image_dir, image)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        predictions = sess.run(result_tensor, 
                               {'DecodeJpeg/contents:0': image_data})
    
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        basename = image.strip('.jpg') + '.txt'
        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)
        filename = os.path.join(scores_dir, basename)
        image_scores = {}
        with open(filename, 'wb') as f:
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                f.write("{l}\t{s}\n".format(l=human_string, s=score))
                image_scores[human_string] = score
        scored_images[image.strip('.jpg')] = image_scores

scored_images_df = pd.DataFrame.from_dict(scored_images)
columns = scored_images_df.columns.tolist().sort(
        key=lambda x: (''.join(i for i in x if not i.isdigit()),
                       safe_int(''.join(i for i in x if i.isdigit()))
                       )
)
scored_images_df.reindex_axis(columns, axis=1)
scored_images_df.to_csv(os.path.join(scores_dir, 'image_scores.csv'))
