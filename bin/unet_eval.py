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
#np.set_printoptions(threshold='nan')
import tensorflow as tf
import quakenet.config as config
import argparse
import os
import time
import glob
import setproctitle
import unet
import fnmatch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# Basic model parameters as external flags.
FLAGS = None


def maybe_save_images(predict_images,images, filenames):
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
            image_array1 = images[i, :,1]
            print (image_array.shape,image_array1.shape)
            indexs=list(range(0,image_array.shape[0])) 
            file_path = os.path.join(FLAGS.output_dir, filenames[i])
            ax = plt.subplot(211)
            plt.plot(indexs,image_array)
            plt.subplot(212, sharex=ax)
            plt.plot(indexs, image_array1)
            plt.savefig(file_path)
            plt.close()
    
def evaluate():
    """
    Eval unet using specified args:
    """
    if FLAGS.events:
        summary_dir =  os.path.join(FLAGS.checkpoint_path,"events")
    while True:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if FLAGS.eval_interval < 0 or ckpt:
            print ('Evaluating model')
            break
        print ('Waiting for training job to save a checkpoint')
        time.sleep(FLAGS.eval_interval)
        
    #data_files, data_size = load_datafiles(FLAGS.tfrecords_prefix)

    setproctitle.setproctitle('quakenet')

    tf.set_random_seed(1234)

    cfg = config.Config()
    cfg.batch_size = FLAGS.batch_size
    cfg.add = 1
    cfg.n_clusters = FLAGS.num_classes
    cfg.n_clusters += 1
    cfg.n_epochs = 1
    model_files = [file for file in os.listdir(FLAGS.checkpoint_path) if
                    fnmatch.fnmatch(file, '*.meta')]
    for model_file in sorted(model_files):
        step = model_file.split(".meta")[0].split("-")[1]
        print (step)
        try:
            model_file = os.path.join(FLAGS.checkpoint_path, model_file)
            # data pipeline for positive and negative examples
            pos_pipeline = dp.DataPipeline(FLAGS.tfrecords_dir, cfg, True)
            #  images:[batch_size, n_channels, n_points]
            images = pos_pipeline.samples
            labels = pos_pipeline.labels
            logits = unet.build_30s(images, FLAGS.num_classes, False)
        
            predicted_images = unet.predict(logits, FLAGS.batch_size, FLAGS.image_size)
        
            accuracy = unet.accuracy(logits, labels)
            loss = unet.loss(logits, labels,FLAGS.weight_decay_rate)
            summary_writer = tf.summary.FileWriter(summary_dir, None)
        
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
        
            sess = tf.Session()
        
            sess.run(init_op)
        
            saver = tf.train.Saver()
        
            #if not tf.gfile.Exists(FLAGS.checkpoint_path + '.meta'):
            if not tf.gfile.Exists(model_file):
                raise ValueError("Can't find checkpoint file")
            else:
                print('[INFO    ]\tFound checkpoint file, restoring model.')
                saver.restore(sess, model_file.split(".meta")[0])
            
            coord = tf.train.Coordinator()
        
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            #metrics = validation_metrics()
            global_accuracy = 0.0
            global_p_accuracy = 0.0
            global_s_accuracy = 0.0
            global_n_accuracy = 0.0
            global_loss = 0.0

            n = 0
            #mean_metrics = {}
            #for key in metrics:
            #    mean_metrics[key] = 0
            #pred_labels = np.empty(1)
            #true_labels = np.empty(1)
        
            try:
                while not coord.should_stop():
                    acc_seg_value,loss_value,predicted_images_value,images_value = sess.run([accuracy,loss,predicted_images,images])
                    accuracy_p_value=acc_seg_value[1][1]
                    accuracy_s_value=acc_seg_value[1][2]
                    accuracy_n_value=acc_seg_value[1][0]
                    #pred_labels = np.append(pred_labels, predicted_images_value)
                    #true_labels = np.append(true_labels, images_value)
                    global_p_accuracy += accuracy_p_value
                    global_s_accuracy += accuracy_s_value
                    global_n_accuracy += accuracy_n_value
                    global_loss += loss_value
                    # print  true_labels
                    #for key in metrics:
                    #    mean_metrics[key] += cfg.batch_size * metrics_[key]
                    filenames_value=[]
                   # for i in range(FLAGS.batch_size):
                   #     filenames_value.append(str(step)+"_"+str(i)+".png")
                    #print (predicted_images_value[:,100:200])
                    if (FLAGS.plot):
                        maybe_save_images(predicted_images_value, images_value,filenames_value)
                    #s='loss = {:.5f} | det. acc. = {:.1f}% | loc. acc. = {:.1f}%'.format(metrics['loss']
                    print('[PROGRESS]\tAccuracy for current batch: |  P. acc. =%.5f| S. acc. =%.5f| '
                          'noise. acc. =%.5f.' % (accuracy_p_value,accuracy_s_value,accuracy_n_value))
                    #n += cfg.batch_size
                    n += 1
                  #  step += 1
                    print (n)
            except KeyboardInterrupt:
                print ('stopping evaluation')
            except tf.errors.OutOfRangeError:
                print ('Evaluation completed ({} epochs).'.format(cfg.n_epochs))
                print ("{} windows seen".format(n))
                #print('[INFO    ]\tDone evaluating in %d steps.' % step)
                if n > 0:
                    loss_value /= n
                    summary = tf.Summary(value=[tf.Summary.Value(tag='loss/val', simple_value=loss_value)])
                    if FLAGS.save_summary:
                        summary_writer.add_summary(summary, global_step=step)
                    global_accuracy /= n
                    global_p_accuracy /= n
                    global_s_accuracy /= n
                    global_n_accuracy /= n
                    summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy/val', simple_value=global_accuracy)])
                    if FLAGS.save_summary:
                        summary_writer.add_summary(summary, global_step=step)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy/val_p', simple_value=global_p_accuracy)])
                    if FLAGS.save_summary:
                        summary_writer.add_summary(summary, global_step=step)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy/val_s', simple_value=global_s_accuracy)])
                    if FLAGS.save_summary:
                        summary_writer.add_summary(summary, global_step=step)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy/val_noise', simple_value=global_n_accuracy)])
                    if FLAGS.save_summary:
                        summary_writer.add_summary(summary, global_step=step)
                    print('[End of evaluation for current epoch]\n\nAccuracy for current epoch:%s | total. acc. =%.5f| P. acc. =%.5f| S. acc. =%.5f| '
                      'noise. acc. =%.5f.' % (step,global_accuracy, global_p_accuracy, global_s_accuracy, global_n_accuracy))
                    print ('Sleeping for {}s'.format(FLAGS.eval_interval))
                    time.sleep(FLAGS.eval_interval)
                summary_writer.flush()
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            tf.reset_default_graph()
            #print('Sleeping for {}s'.format(FLAGS.eval_interval))
            #time.sleep(FLAGS.eval_interval)
        finally:
            print  ('joining data threads')

            coord = tf.train.Coordinator()
            coord.request_stop()

    #pred_labels = pred_labels[1::]
    #true_labels = true_labels[1::]
    #print  ("---Confusion Matrix----")
    #print (confusion_matrix(true_labels, pred_labels))
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(_):
    """
    Run unet prediction on input tfrecords
    """
    if FLAGS.output_dir is not None:
        if not tf.gfile.Exists(FLAGS.output_dir):
            print('[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(FLAGS.output_dir)
        
    evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Eval Unet on given tfrecords.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'training')
    parser.add_argument('--checkpoint_path', help = 'Path of checkpoint to restore. (Ex: ../Datasets/checkpoints/unet.ckpt-80000)')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='sleep time between evaluations')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 3)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 3001)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 2)
    parser.add_argument('--events', action='store_true',
                        help='pass this flag if evaluate acc on events')
    parser.add_argument('--weight_decay_rate', help='Weight decay rate', type=float, default=0.0005)
    parser.add_argument('--save_summary',type=bool,default=True,
                        help='True to save summary in tensorboard')
    parser.add_argument('--output_dir', help = 'Output directory for the prediction files. If this is not set then predictions will not be saved')
    parser.add_argument("--plot", action="store_true",help="pass flag to plot detected events in output")
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
