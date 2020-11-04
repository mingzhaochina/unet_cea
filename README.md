# unet_cea
Using Unet to train model based on wenchuan aftershocks and automatically detect p,s phases
============= 

https://doi.org/10.5281/zenodo.4243864

This is the unet code used in the paper:
赵明，陈石，房立华，David A Yuen. 2019. 基于U形全卷积神经网络的震相识别与到时拾取方法研究. 地球物理学报，待刊.

And,it is developed from ConvNetQuake :
Perol., T, M. Gharbi and M. Denolle. Convolutional Neural Network for Earthquake detection and location. [preprint arXiv:1702.02073](https://arxiv.org/abs/1702.02073), 2017.

The u-net structure

![The u-net](./fig1.jpg)

The example of labeled sample:

![labeled sample](./fig2.jpg)

Some detect results

![Detections](./fig3.jpg)

Some train snapshots

![snapshots](./20190414211527.png)

## Installation
* `export PYTHONPATH=`pwd`:$PYTHONPATH`

* Install dependencies: `conda env create -f  unet_cea.yaml`

## create dataset

First you need a phase file for doing such thing:

./bin/create_dataset_events_unet.py --stream_dir waveform --catalog ps_picks_capital.csv --output_dir train_30s_unet --plot True --window_size 30

the ps_picks_capital.csv has a format like this:

,ev_dir,ev_id,ev_start,p_pick,s_pick,stname,time_diff,utc_timestamp,utc_timestamp_s

0,HE.200901010316.0001,1,2009-01-01T03:16:45.759000Z,2009-01-01T03:16:49.219000Z,2009-01-01T03:16:50.599000Z,CC.XBZ,1.3799998760223389,1230751009.219,1230751010.599

1,HE.200901010316.0001,1,2009-01-01T03:16:45.759000Z,2009-01-01T03:16:51.779000Z,2009-01-01T03:16:56.459000Z,BU.LQS,4.680000066757202,1230751011.779,1230751016.459

2,HE.200901010316.0001,1,2009-01-01T03:16:45.759000Z,2009-01-01T03:16:52.129000Z,2009-01-01T03:16:55.599000Z,BJ.SSL,3.4700000286102295,1230751012.129,1230751015.599

....................................................................................................................................................................

## Train(just a small data to show the code really work ^_^)

python ./bin/unet_train.py --tfrecords_dir data/train/  --checkpoint_dir model

## validate
python ./bin/unet_eval.py --tfrecords_dir data/test --checkpoint_path output/unet_capital/  --batch_size 1000 --output_dir output/unet --events

## Tensorboard for real-time monitor

tensorboard --logdir model

## Trained model
An trained model on Chinese Metropolitian Network(178 stations,266350 samples),thanks to  Hebei Earthquake Administration for providing the catalogs and high accuracy manual picks 

The directory `unet_capital`

Use new test data(not used in the train and validate process) to check the preformance of the model:

python ./bin/unet_eval_from_tfrecords.py --tfrecords_dir detection --checkpoint_path ./unet_capital/unet.ckpt-585000 --batch_size 8 --output_dir output

or you can test with your own data (mseed)

可以直接用以下程序，载入你自己的mseed格式波形，进行实时检测

python ./bin/unet_eval_from_stream.py --stream_path ./mseed/  --checkpoint_path unet_capital/unet.ckpt-590000 --batch_size 8 --output_dir output/predict_from_stream --plot

(more to come) 
