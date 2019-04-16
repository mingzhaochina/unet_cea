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
from quakenet.data_io import load_stream
import numpy as np
import pandas as pd
import tensorflow as tf
import quakenet.config as config
import argparse
import os,shutil
import time
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
import tqdm
import glob
import setproctitle
import unet
import fnmatch
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
def tree(top):
     #path,folder list,file list
    for path, names, fnames in os.walk(top):
        for fname in fnames:
            yield os.path.join(path, fname)
def fetch_window_data(stream,j):
    """fetch data from a stream window and dump in np array"""
    cfg = config.Config()
    data = np.empty((cfg.win_size, j))
    for i in range(j):
        data[:, i] = stream[i].data.astype(np.float32)
    data = np.expand_dims(data, 0)
    return data
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result
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
def preprocess_stream(stream):
    stream = stream.detrend('constant')
    ##add by mingzhao,2017/12/2
    stream =stream.filter('bandpass', freqmin=0.5, freqmax=20)
    ##########
    return stream

def data_is_complete(stream):
    """Returns True if there is 1001*3 points in win"""
    cfg = config.Config()
    try:
        data_size = len(stream[0].data) + len(stream[1].data) + len(stream[2].data)
    except:
        data_size = 0
    data_lenth=int(cfg.win_size)*3
    if data_size == data_lenth:
        return True
    else:
        return False
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

    #data_files, data_size = load_datafiles(FLAGS.tfrecords_prefix)

    setproctitle.setproctitle('quakenet')

    tf.set_random_seed(1234)

    cfg = config.Config()
    cfg.batch_size = 1
    cfg.add = 1
    cfg.n_clusters = FLAGS.num_classes
    cfg.n_clusters += 1

    # stream data with a placeholder
    samples = {
        'data': tf.placeholder(tf.float32,
                                   shape=(cfg.batch_size, cfg.win_size, 3),
                                   name='input_data')}
    stream_path = FLAGS.stream_path
    try:
        #stream_files = [file for file in os.listdir(stream_path) if
        #                fnmatch.fnmatch(file, '*.mseed')]
        stream_files = [file for file in tree(stream_path) if
                        fnmatch.fnmatch(file, '*.mseed')]
    except:
        stream_files = os.path.split(stream_path)[-1]
        print ("stream_files",stream_files)
    #data_files, data_size = load_datafiles(stream_path)
    n_events = 0
    time_start = time.time()
    print(" + Loading stream files {}".format(stream_files))

    events_dic = {"slice_start_time":[],
                  "P_pick": [],
                  "stname": [],
                  "utc_timestamp_p": [],
                  "utc_timestamp_s": [],
                  "S_pick": []}
    with tf.Session() as sess:

        logits = unet.build_30s(samples['data'], FLAGS.num_classes, False)
        time_start = time.time()
        catalog_name = "PS_pick_blocks.csv"
        output_catalog = os.path.join(FLAGS.output_dir, catalog_name)
        print('Catalog created to store events', output_catalog)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)
        saver = tf.train.Saver()

        if not tf.gfile.Exists(FLAGS.checkpoint_path + '.meta'):
            raise ValueError("Can't find checkpoint file")
        else:
            print('[INFO    ]\tFound checkpoint file, restoring model.')
            saver.restore(sess, FLAGS.checkpoint_path)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for stream_file in stream_files:
            #stream_path1 = os.path.join(stream_path, stream_file)
            print (" + Loading stream {}".format(stream_file))
            #stream = load_stream(stream_path1)
            stream = load_stream(stream_file)
            stream =stream.normalize()
            #print stream[0],stream[1].stats
            print (" + Preprocess stream")
           # stream = preprocess_stream(stream)
            print (" -- Stream is ready, starting detection")
            try:
                #lists = [0]
                lists = np.arange(0,30,10)
                for i in lists:
                    win_gen = stream.slide(window_length=FLAGS.window_size, step=FLAGS.window_step,offset=i,
                                                                                 include_partial_windows=False)
                    #print(win_gen)
                    for idx, win in enumerate(win_gen):
                        #win.resample(10)
                        if data_is_complete(win):
                            predicted_images = unet.predict(logits, cfg.batch_size, FLAGS.image_size)
                            to_fetch = [predicted_images,samples['data']]

                            # Feed window and fake cluster_id (needed by the net) but
                            # will be predicted
                            feed_dict = {samples['data']: fetch_window_data(win.copy().normalize(),3)}
                            #samples_data=fetch_window_data(win.copy().normalize()
                            predicted_images_value, images_value = sess.run(to_fetch,feed_dict)
                            clusters_p =np.where(predicted_images_value[0,:]==1)
                            clusters_s = np.where(predicted_images_value[0,:] == 2)
                            p_boxes = group_consecutives(clusters_p[0])
                            s_boxes = group_consecutives(clusters_s[0])
                            tp=[]
                            ts=[]
                            tpstamp = []
                            tsstamp = []
                            if len(p_boxes) > 1:
                                for ip in range(len(p_boxes)):
                                    #print (len(p_boxes),p_boxes,p_boxes[ip])
                                    tpmean=float(min(p_boxes[ip])/200.00+max(p_boxes[ip])/200.00)
                                    tp.append(tpmean)
                                    tpstamp=UTCDateTime(win[0].stats.starttime+tpmean).timestamp
                            if len(s_boxes) > 1:
                                for iss in range(len(s_boxes)):
                                    tsmean=float(min(s_boxes[iss])/200.00+max(s_boxes[iss])/200.00)
                                    ts.append(tsmean)
                                    tsstamp=UTCDateTime(win[0].stats.starttime+tsmean).timestamp
                            if len(p_boxes) > 1 or len(s_boxes) > 1:
                                events_dic["slice_start_time"].append(win[0].stats.starttime)
                                events_dic["stname"].append(win[0].stats.station)
                                events_dic["P_pick"].append(tp)
                                events_dic["S_pick"].append(ts)
                                events_dic["utc_timestamp_p"].append(tpstamp)
                                events_dic["utc_timestamp_s"].append(tsstamp)
                            #print (p_boxes,s_boxes)
                            win_filtered = win.copy()
                            lab = win_filtered[2].copy()
                            lab.stats.channel = "LAB"
                            # lab =win[0].copy()

                            print("predicted_images_value", predicted_images_value.shape)
                            lab.data[...] = predicted_images_value[0, :]
                            win_filtered += lab
                            if FLAGS.save_sac:
                                output_sac=os.path.join(FLAGS.output_dir, "sac",
                                "{}_{}.sac".format(win_filtered[0].stats.station,
                                str(win_filtered[0].stats.starttime).replace(':', '_')))
                                print (output_sac,win_filtered)
                                win_filtered.write(output_sac,format="SAC")
                            if FLAGS.plot:
                                win_filtered.plot(outfile=os.path.join(FLAGS.output_dir, "viz",
                                                                       "{}_{}.png".format(win_filtered[0].stats.station,
                                                                        str(win_filtered[0].stats.starttime).replace(':', '_'))))
                            # Wait for threads to finish.
                            coord.join(threads)
            except KeyboardInterrupt:
                print ('Interrupted at time {}.'.format(win[0].stats.starttime))
                print ("processed {} windows, found {} events".format(idx+1,n_events))
                print ("Run time: ", time.time() - time_start)
        df = pd.DataFrame.from_dict(events_dic)
        df.to_csv(output_catalog)

    print ("Run time: ", time.time() - time_start)
def main(_):
    """
    Run unet prediction on input tfrecords
    """

    if FLAGS.output_dir is not None:
        if not tf.gfile.Exists(FLAGS.output_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(FLAGS.output_dir)
    if FLAGS.plot:
        viz_dir=os.path.join(FLAGS.output_dir, "viz")
        if not tf.gfile.Exists(viz_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(viz_dir)
    if FLAGS.save_sac:
        sac_dir =os.path.join(FLAGS.output_dir,"sac")
        if not tf.gfile.Exists(sac_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(sac_dir)
    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Unet on given tfrecords.')
    parser.add_argument('--stream_path', help='Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help='Tfrecords prefix', default='mseed')
    parser.add_argument('--checkpoint_path',
                        help='Path of checkpoint to restore. (Ex: ../Datasets/checkpoints/unet.ckpt-80000)')
    parser.add_argument("--window_size",type=int,default=30,
                        help="size of the window to analyze")
    parser.add_argument("--window_step",type=int,default=31,
                        help="step between windows to analyze")
    parser.add_argument('--num_classes', help='Number of segmentation labels', type=int, default=3)
    parser.add_argument('--image_size', help='Target image size (resize)', type=int, default=3001)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--output_dir',
                        help='Output directory for the prediction files. If this is not set then predictions will not be saved')
    parser.add_argument("--plot", action="store_true",
                        help="pass flag to plot detected events in output")
    parser.add_argument("--save_sac", action="store_true",
                        help="pass flag to plot detected events in output")

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
