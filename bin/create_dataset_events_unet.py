#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : create_dataset_events.py
# Creation Date : 05-12-2016
# Last Modified : Fri Jan  6 15:04:54 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""Creates tfrecords dataset of events trace and their cluster_ids.
This is done by loading a dir of .mseed and one catalog with the
time stamps of the events and their cluster_id
e.g.,
./bin/preprocess/create_dataset_events.py \
--stream_dir data/streams \
--catalog data/50_clusters/catalog_with_cluster_ids.csv\
--output_dir data/50_clusters/tfrecords
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from quakenet.data_pipeline_unet import DataWriter
import tensorflow as tf
from obspy.core import read,Stream
from quakenet.data_io import load_catalog
from obspy.core.utcdatetime import UTCDateTime
import fnmatch,math
import json

flags = tf.flags
flags.DEFINE_string('stream_dir', None,
                    'path to the directory of streams to preprocess.')
flags.DEFINE_string(
    'catalog', None, 'path to the events catalog to use as labels.')
flags.DEFINE_string('output_dir', None,
                    'path to the directory in which the tfrecords are saved')
flags.DEFINE_bool("plot", False,
                  "If we want the event traces to be plotted")
flags.DEFINE_bool("augmentation", True,
                  "If we want the event traces to be plotted")
flags.DEFINE_float(
    'window_size', 10, 'size of the window samples (in seconds)')
flags.DEFINE_float('v_mean', 5.0, 'mean velocity')
flags.DEFINE_boolean("save_mseed",False,
                     "save the windows in mseed format")
FLAGS = flags.FLAGS

def Bandpass(data, flp, fhp, dt, n):
    #  Butterworth Acausal Bandpass Filter
    #
    #   Syntax:
    #          x_f = bandpass(c, flp, fhi, dt, n)
    #
    #   Input:
    #            x = input time series
    #          flp = low-pass corner frequency in Hz
    #          fhi = high-pass corner frequency in Hz
    #           dt = sampling interval in second
    #            n = order
    #
    #   Output:
    #        x_f = bandpass filtered signal
    fs = 1/dt              # Nyquist frequency
    b, a = Butter_Bandpass(flp, fhp, fs, order=n)
    x_f = filtfilt(b, a, data, padlen = 3*(max(len(b), len(a)) - 1))
    return x_f

def Butter_Bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

##add by mingzhao,2017/12/2
def filter_small_ampitude(st_event,n_samples):
    a_e = 1.0 * len(filter(lambda x: -5.0<= x <= -5.0, st_event[0].data)) / n_samples
    a_n = 1.0 * len(filter(lambda x: -5.0<= x <= -5.0, st_event[1].data)) / n_samples
    a_z = 1.0 * len(filter(lambda x: -5.0<= x <= -5.0, st_event[2].data)) / n_samples
   # print (87,a_e,a_n,a_z)
    return a_e,a_n,a_z

def remove_repeat(st_event,n_samples,j):
    dic={}
    a=[]
    for i in range(j):
        for item in st_event[i].data:
            if item in dic.keys():
                dic[item]+=1
            else:
                dic[item]=1
        mm=max(dic.values())
        a.append(1.0 * mm / n_samples)
    #print (a)
    return a
def preprocess_stream(stream):
    stream = stream.detrend('constant')
    ##add by mingzhao,2017/12/2
    #stream =stream.filter('bandpass', freqmin=0.5, freqmax=20)
    ##########
    return stream

#def draw_bounding_boxes:
#    img_data=tf.image.resize_images(img_data,180,267,method=1)
#    batched = tf.extend_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
#    boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
#    result = tf.image.draw_bounding_boxes(batched,boxes)
def filter_catalog(cat,stream_file):
    import re
    # Filter around Guthrie sequence
    #stlog = pd.read_csv('/home/zm/obspy/station_latlon.csv')
    #m2 = re.search(stream_file.split(".")[1] ,stlog.name)

    #cat = cat[(cat.latitude > 35.7) & (cat.latitude < 36)
    #          & (cat.longitude > -97.6) & (cat.longitude < -97.2)]
    #match stream_file,so that all the name of the stream_file contains the key word  will be matched,2017/12/07
    m1=re.match('(\D+)', stream_file.split(".")[1].ljust(4))
    print m1.group(),1
    cat = cat[(cat.stname == '{0:<4}'.format(m1.group()))]
   #           (cat.stname == str(stream_file)[:-1]))]
    return cat


def get_travel_time(catalog):
    """Find the time between origin and propagation"""
    v_mean = FLAGS.v_mean
    coordinates = [(lat, lon, depth) for (lat, lon, depth) in zip(catalog.latitude,
                                                                  catalog.longitude,
                                                                  catalog.depth)]
    distances_to_station = [distance_to_station(lat, lon, depth)
                            for (lat, lon, depth) in coordinates]
    travel_time = [distance/v_mean for distance in distances_to_station]
    return travel_time

def write_json(metadata,output_metadata):
    with open(output_metadata, 'w') as outfile:
        json.dump(metadata, outfile)

def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
def main(_):
    stream_dirs = [file for file in os.listdir(FLAGS.stream_dir)]

    print "List of streams to anlayze", stream_dirs

    # Create dir to store tfrecords
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Dictionary of nb of events per tfrecords
    metadata = {}
    output_metadata = os.path.join(FLAGS.output_dir,"metadata.json")

    # Load Catalog

    #evlog = load_catalog(FLAGS.catalog)
    #print ("+ Loading Catalog:",evlog)
    for stream_dir in stream_dirs:

        #cat = evlog[(evlog.stname == stream_dir)]
        #print cat
        # Load stream
        stream_path = os.path.join(FLAGS.stream_dir, stream_dir,"local")
        waveforms=read(os.path.join(stream_path, '*.SAC'))
        waveforms.sort
        #print waveforms[0]
        output_name = stream_dir + ".tfrecords"
        output_path = os.path.join(FLAGS.output_dir, output_name)
        writer = DataWriter(output_path)
        print("+ Creating tfrecords for {} events".format(len(waveforms)))
        for i in range(0, len(waveforms), 3):
            print "+ Loading Stream\n {}\n{}\n{}".format(waveforms[i],waveforms[i+1],waveforms[i+2])
            #stream_filepath = os.path.join(stream_path, stream_file)
            #stream = read(stream_filepath)
            #print '+ Preprocessing stream',stream
        #stream = preprocess_stream(stream)

        # Filter catalog according to the loaded stream

            start_date = waveforms[i].stats.starttime
            end_date = waveforms[i+2].stats.endtime
            print("-- Start Date={}, End Date={}".format(start_date, end_date))
            x = np.random.randint(0, 4)
            st_event = waveforms[i:i+3].resample(100).trim(start_date+x, start_date+x+FLAGS.window_size,pad=True, fill_value=0.0).copy()
            #st_event.resample(100)
            print (st_event)
            n_samples = len(st_event[0].data)
            sample_rate = st_event[0].stats.sampling_rate
            n_pts = sample_rate * FLAGS.window_size + 1
            cluster_id_p = 5-x
            cluster_id_s = end_date - start_date-x-15
            if cluster_id_s>=30:
                continue
            assert n_pts == n_samples, "n_pts and n_samples are not the same"
        # Write event waveforms and cluster_id in .tfrecords

            # for p picks
            # u=0
            # label = np.zeros((n_samples), dtype=np.float32)
            label_obj = st_event.copy()

            label_obj[0].data[...] = 1
            label_obj[1].data[...] = 0
            label_obj[2].data[...] = 0
            u1 = cluster_id_p * sample_rate  # mean value miu
            lower = int(u1 - 0.5 * sample_rate)
            upper = int(u1 + 0.5 * sample_rate)
            label_obj[1].data[lower:upper] = 1
            # label_obj.data[int(u1 - 0.5 * sample_rate):int(u1 + 0.5 * sample_rate)] = 1
            # y_sig = np.random.normal(u1, sig, n_samples )
            # for s pick
            u2 = cluster_id_s * sample_rate  # mean value miu

            lower2, upper2 = int(u2 - sample_rate), int(u2 + sample_rate)
            try:
                label_obj[2].data[lower2:upper2] = 2
                # label_obj.data[int(u2 - sample_rate):int(u2 + sample_rate)] =2
            except:
                nnn = int(n_samples) - int(u2 + sample_rate)
                print  (nnn, n_samples)
                label_obj[2].data[lower2:n_samples] = 2
            label_obj.normalize()
            label_obj[0].data = label_obj[0].data - label_obj[1].data - label_obj[2].data
            # label_obj.data[int(u2 - sample_rate):n_samples] = 2
            writer.write(st_event.copy().normalize(), label_obj)
            if FLAGS.save_mseed:
                output_label = "{}_{}.mseed".format(st_event[0].stats.station,
                                                    str(st_event[0].stats.starttime).replace(':', '_'))

                output_mseed_dir = os.path.join(FLAGS.output_dir, "mseed")
                if not os.path.exists(output_mseed_dir):
                    os.makedirs(output_mseed_dir)
                output_mseed = os.path.join(output_mseed_dir, output_label)
                st_event.write(output_mseed, format="MSEED")

            # Plot events
            if FLAGS.plot:
                #import matplotlib
                #matplotlib.use('Agg')

                # from obspy.core import Stream
                traces = Stream()
                traces += st_event[0].filter('bandpass', freqmin=0.5, freqmax=20)
                traces += label_obj
                # print traces
                viz_dir = os.path.join(
                    FLAGS.output_dir, "viz",stream_dir)
                if not os.path.exists(viz_dir):
                    os.makedirs(viz_dir)
                traces.normalize().plot(outfile=os.path.join(viz_dir,
                                                             ####changed at 2017/11/25,use max cluster_prob instead of cluster_id
                                                             #                "event_{}_cluster_{}.png".format(idx,cluster_id)))
                                                             "event_{}_{}.png".format(
                                                                 st_event[0].stats.station,
                                                                 str(st_event[0].stats.starttime).replace(
                                                                     ':', '_'))))
        # Cleanup writer
        print("Number of events written={}".format(writer._written))
        writer.close()
        # Write metadata
        metadata[stream_dir] = writer._written
        write_json(metadata, output_metadata)


if __name__ == "__main__":
    tf.app.run()
