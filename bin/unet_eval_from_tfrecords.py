# ============================================================== #
#                         Fusnet eval                            #
#                                                                #
#                                                                #
# Eval fusnet with processed dataset in tfrecords format         #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

from __future__ import print_function

import quakenet.data_pipeline_unet as dp
import numpy as np
import tensorflow as tf
import quakenet.config as config
import argparse
import os
import time
import glob
import setproctitle
import unet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Basic model parameters as external flags.
FLAGS = None


def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '*.%s' % type)
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size


def maybe_save_images(predict_images, images, filenames):
    """
    Save images to disk
    -------------
    Args:
        images: numpy array     [batch_size, image_size, image_size]
        filenames: numpy string array, filenames corresponding to the images   [batch_size]
    """
    if FLAGS.output_dir is not None:
        batch_size = predict_images.shape[0]
        for i in xrange(batch_size):
            image_array = predict_images[i, :]
            image_array1 = images[i, :, 1]
            print(image_array.shape, image_array1.shape)
            indexs = list(range(0, image_array.shape[0]))
            file_path = os.path.join(FLAGS.output_dir, filenames[i])
            ax = plt.subplot(211)
            plt.plot(indexs, image_array)
            plt.subplot(212, sharex=ax)
            plt.plot(indexs, image_array1)
            plt.savefig(file_path)
            plt.close()


def evaluate():
    """
    Eval unet using specified args:
    """


    setproctitle.setproctitle('quakenet')

    tf.set_random_seed(1234)

    cfg = config.Config()
    cfg.batch_size = FLAGS.batch_size
    cfg.add = 1
    cfg.n_clusters = FLAGS.num_classes
    cfg.n_clusters += 1

    # data pipeline for positive and negative examples
    pos_pipeline = dp.DataPipeline(FLAGS.tfrecords_dir, cfg, True)
    #  images:[batch_size, n_channels, n_points]
    images = pos_pipeline.samples
    labels = pos_pipeline.labels
    print("images", images,labels)
    logits = unet.build_30s(images, FLAGS.num_classes, False)

    predicted_images = unet.predict(logits, FLAGS.batch_size, FLAGS.image_size)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    saver = tf.train.Saver()

    if not tf.gfile.Exists(FLAGS.checkpoint_path + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        print('[INFO    ]\tFound checkpoint file, restoring model.')
        saver.restore(sess, FLAGS.checkpoint_path)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    global_accuracy = 0.0

    step = 0

    try:
        while not coord.should_stop():
            print(predicted_images, images)
            predicted_images_value, images_value = sess.run([predicted_images, images])
            print (predicted_images_value, images_value)
            filenames_value = []
            for i in range(FLAGS.batch_size):
                filenames_value.append(str(step) + "_" + str(i) + ".png")
            #print (predicted_images_value[:,100:200])
            maybe_save_images(predicted_images_value, images_value, filenames_value)
            step += 1

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone evaluating in %d steps.' % step)

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(_):
    """
    Run unet prediction on input tfrecords
    """

    if FLAGS.output_dir is not None:
        if not tf.gfile.Exists(FLAGS.output_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(FLAGS.output_dir)

    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Unet on given tfrecords.')
    parser.add_argument('--tfrecords_dir', help='Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help='Tfrecords prefix', default='training')
    parser.add_argument('--checkpoint_path',
                        help='Path of checkpoint to restore. (Ex: ../Datasets/checkpoints/unet.ckpt-80000)')
    parser.add_argument('--num_classes', help='Number of segmentation labels', type=int, default=3)
    parser.add_argument('--image_size', help='Target image size (resize)', type=int, default=3001)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--output_dir',
                        help='Output directory for the prediction files. If this is not set then predictions will not be saved')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
