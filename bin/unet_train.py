# ============================================================== #
#                          Unet train                            #
#                                                                #
#                                                                #
# Train fusnet with processed dataset in tfrecords format        #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

#from __future__ import print_function
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


def train():
    """
    Train unet using specified args:
    """

    setproctitle.setproctitle('quakenet')

    tf.set_random_seed(1234)


    cfg = config.Config()
    cfg.batch_size = FLAGS.batch_size
    cfg.add = 1
    cfg.n_clusters = FLAGS.num_classes
    cfg.n_clusters += 1

    # data pipeline for positive and negative examples

    data_files, data_size = load_datafiles(FLAGS.tfrecords_prefix)
    print data_files,data_size
    pos_pipeline = dp.DataPipeline(FLAGS.tfrecords_dir, cfg, True)
#  images:[batch_size, n_channels, n_points]
    images= pos_pipeline.samples
    labels = pos_pipeline.labels
    print "1111111labels,images",labels,images
    logits = unet.build_30s(images, FLAGS.num_classes, True)
    print "logits111",logits,labels
    accuarcy = unet.accuracy(logits, labels)
    #print "accuarcy,recall,f1", accuarcy,recall,f1
    print "accuarcy,recall,f1", accuarcy
    #load class weights if available
    if FLAGS.class_weights :
        weights = [1.0,50.0,50.0]
        class_weight_tensor = tf.constant(weights, dtype=tf.float32, shape=[FLAGS.num_classes, 1])
    else:
        class_weight_tensor = None
    loss = unet.loss(logits, labels, FLAGS.weight_decay_rate,class_weight_tensor)
    print "loss",loss
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = unet.train(loss, FLAGS.learning_rate, FLAGS.learning_rate_decay_steps, FLAGS.learning_rate_decay_rate, global_step)
    #print "train_op",train_op

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()
    session_manager = tf.train.SessionManager(local_init_op = tf.local_variables_initializer())
    sess = session_manager.prepare_session("", init_op = init_op, saver = saver, checkpoint_dir = FLAGS.checkpoint_dir)

    writer = tf.summary.FileWriter(FLAGS.checkpoint_dir + "/train_logs", sess.graph)

    merged = tf.summary.merge_all()

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    start_time = time.time()

    try:
        while not coord.should_stop():

            step = tf.train.global_step(sess, global_step)
            _, loss_value, summary = sess.run([train_op, loss, merged])
            #print loss_value
            writer.add_summary(summary, step)
            if step % 1000 == 0:
                acc_seg_value = sess.run([accuarcy])
                #print "acc_seg_value:",acc_seg_value,acc_seg_value[0],acc_seg_value[0][1],acc_seg_value[0][1][0]
                epoch = step * FLAGS.batch_size / data_size
                #print epoch
                duration = time.time() - start_time
                #print step,duration
                start_time = time.time()
                #print('[PROGRESS]\tEpoch %d | Step %d | loss = %.2f | total. acc. = %.2f | P. acc. =  %.3f \
                #      | S. acc. =  %.3f | N. acc. =  %.3f | dur. = (%.3f sec)'\
                #      % (epoch, step, loss_value, acc_seg_value[0][1][0],acc_seg_value[0][1][1], acc_seg_value[0][1][2],\
                #         acc_seg_value[0][3],duration))

                print('[PROGRESS]\tEpoch %d | Step %d | loss = %.2f | P. acc. =  %.3f \
                      | S. acc. =  %.3f | N. acc. =  %.3f | dur. = (%.3f sec)'\
                      % (epoch, step, loss_value, acc_seg_value[0][1][1],acc_seg_value[0][1][2], acc_seg_value[0][1][0],\
                         duration))
            if step % 5000 == 0:
                print('[PROGRESS]\tSaving checkpoint')
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'unet.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    writer.close()
    sess.close()


def main(_):
    """
    Download processed dataset if missing & train
    """

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        print('[INFO    ]\tCheckpoint directory does not exist, creating directory: ' + os.path.abspath(FLAGS.checkpoint_dir))
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train Unet on given tfrecords directory.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'tfrecords')
    parser.add_argument('--checkpoint_dir', help = 'Checkpoints directory')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 3)
    parser.add_argument('--class_weights', help = 'Weight per class for weighted loss.  [num_classes]',type=bool,default=True)
    parser.add_argument('--learning_rate', help = 'Learning rate', type = float, default = 1e-4)
    parser.add_argument('--learning_rate_decay_steps', help = 'Learning rate decay steps', type = int, default = 10000)
    parser.add_argument('--learning_rate_decay_rate', help = 'Learning rate decay rate', type = float, default = 0.9)
    parser.add_argument('--weight_decay_rate', help = 'Weight decay rate', type = float, default = 0.0005)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 4)
    parser.add_argument('--num_epochs', help = 'Number of epochs', type = int, default = 15)

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
