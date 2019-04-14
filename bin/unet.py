# ============================================================== #
#                            U-net                               #
#                                                                #
#                                                                #
# Unet tensorflow implementation                                 #
#                                                                #
# Author: Karim Tarek
# 
# Add new accuracy strategy
#
# ============================================================== #

import tensorflow as tf
#import tflib.model
#import tflib.layers as layers
import layers
from keras.layers import Cropping1D
def build_30s(color_inputs, num_classes, is_training):
    """
    Build unet network:
    ----------
    Args:
        color_inputs: Tensor, [batch_size, length, 3]
        num_classes: Integer, number of segmentation (annotation) labels
        is_training: Boolean, in training mode or not (for dropout & bn)
    Returns:
        logits: Tensor, predicted annotated image flattened 
                              [batch_size * length,  num_classes]
    """

    dropout_keep_prob = tf.where(is_training, 0.2, 1.0)

    # Encoder Section
    # Block 1
   # color_conv1_1 = layers.conv_btn(color_inputs,  [3, 3], 64, 'conv1_1', is_training = is_training)

    color_conv1_1 = layers.conv_btn1(color_inputs, 3, 32, 'conv1_1', is_training=is_training)
    #layers.conv1(current_layer, c, ksize, stride=2, scope='conv{}'.format(i + 1), padding='SAME')
    color_conv1_2 = layers.conv_btn1(color_conv1_1, 3, 32, 'conv1_2', is_training = is_training)
    color_pool1   = layers.maxpool(color_conv1_2, 4,  'pool1')

    # Block 2
    color_conv2_1 = layers.conv_btn1(color_pool1 ,   3, 32, 'conv2_1', is_training = is_training)
    color_conv2_2 = layers.conv_btn1(color_conv2_1, 3, 32, 'conv2_2', is_training = is_training)
    color_pool2   = layers.maxpool(color_conv2_2, 4,   'pool2')
    # Block 3
    color_conv3_1 = layers.conv_btn1(color_pool2,   3, 64, 'conv3_1', is_training = is_training)
    color_conv3_2 = layers.conv_btn1(color_conv3_1, 3, 64, 'conv3_2', is_training = is_training)
    color_pool3   = layers.maxpool(color_conv3_2, 4,   'pool3')
    color_drop3   = layers.dropout(color_pool3, dropout_keep_prob, 'drop3')
    # Block 4
    color_conv4_1 = layers.conv_btn1(color_drop3,   3, 64, 'conv4_1', is_training = is_training)
    color_conv4_2 = layers.conv_btn1(color_conv4_1, 3, 64, 'conv4_2', is_training = is_training)
    color_pool4   = layers.maxpool(color_conv4_2, 4,   'pool4')
    color_drop4   = layers.dropout(color_pool4, dropout_keep_prob, 'drop4')

    # Block 5
    color_conv5_1 = layers.conv_btn1(color_drop4,   3, 128, 'conv5_1', is_training = is_training)
    color_conv5_2 = layers.conv_btn1(color_conv5_1, 3, 128, 'conv5_2', is_training = is_training)
    color_drop5   = layers.dropout(color_conv5_2, dropout_keep_prob, 'drop5')

    # Decoder Section
    # Block 1

    upsample61     = layers.deconv_upsample(color_drop5, 4,  'upsample6')
    upsample61 = Cropping1D(cropping=((0, 1)))(upsample61)
    concat6       = layers.concat(upsample61, color_conv4_2, 'concat6')
    color_conv6_1 = layers.conv_btn1(concat6,       3, 128, 'conv6_1', is_training = is_training)
   # color_conv6_2 = layers.conv_btn1(color_conv6_1, 6, 128, 'conv6_2', is_training = is_training)
    color_drop6   = layers.dropout(color_conv6_1, dropout_keep_prob, 'drop6')
    # Block 2
    upsample7     = layers.deconv_upsample(color_drop6, 4,  'upsample7')
   # upsample7 = Cropping1D(cropping=((0, 1)))(upsample7)
    concat7       = layers.concat(upsample7, color_conv3_2, 'concat7')
    color_conv7_1 = layers.conv_btn1(concat7,       3, 64, 'conv7_1', is_training = is_training)
   # color_conv7_2 = layers.conv_btn1(color_conv7_1, 6, 64, 'conv7_1', is_training = is_training)
    color_drop7   = layers.dropout(color_conv7_1, dropout_keep_prob, 'drop7')

    # Block 3
    upsample81     = layers.deconv_upsample(color_drop7, 4,  'upsample8')
    upsample81 = Cropping1D(cropping=((0, 1)))(upsample81)
    concat8       = layers.concat(upsample81, color_conv2_2, 'concat8')
    color_conv8_1 = layers.conv_btn1(concat8,       3, 32, 'conv8_1', is_training = is_training)
   # color_conv8_2 = layers.conv_btn1(color_conv8_1, 3, 32, 'conv8_1', is_training = is_training)

    # Block 4
    upsample91     = layers.deconv_upsample(color_conv8_1, 4, 'upsample9')
    upsample91 = Cropping1D(cropping=((1, 2)))(upsample91)
    concat9       = layers.concat(upsample91, color_conv1_2,  'concat9')
    color_conv9_1 = layers.conv_btn1(concat9,       3, 32,   'conv9_1', is_training = is_training)
   # color_conv9_2 = layers.conv_btn1(color_conv9_1, 3, 32,   'conv9_1', is_training = is_training)

    # Block 5
    score  = layers.conv(color_conv9_1, 1, num_classes, 'score', activation_fn = None)
    logits = tf.reshape(score, (-1, num_classes))
    return logits

def build_40s(color_inputs, num_classes, is_training):
    """
    Build unet network:
    ----------
    Args:
        color_inputs: Tensor, [batch_size, length, 3]
        num_classes: Integer, number of segmentation (annotation) labels
        is_training: Boolean, in training mode or not (for dropout & bn)
    Returns:
        logits: Tensor, predicted annotated image flattened
                              [batch_size * length,  num_classes]
    """

    dropout_keep_prob = tf.where(is_training, 0.2, 1.0)

    # Encoder Section
    # Block 1
   # color_conv1_1 = layers.conv_btn(color_inputs,  [3, 3], 64, 'conv1_1', is_training = is_training)

    color_conv1_1 = layers.conv_btn1(color_inputs, 3, 32, 'conv1_1', is_training=is_training)
    print 011,color_conv1_1
    #layers.conv1(current_layer, c, ksize, stride=2, scope='conv{}'.format(i + 1), padding='SAME')
    color_conv1_2 = layers.conv_btn1(color_conv1_1, 3, 32, 'conv1_2', is_training = is_training)
    print 012,color_conv1_2
    color_pool1   = layers.maxpool(color_conv1_2, 4,  'pool1')

    # Block 2
    color_conv2_1 = layers.conv_btn1(color_pool1 ,   3, 32, 'conv2_1', is_training = is_training)
    print 021,color_conv2_1
    color_conv2_2 = layers.conv_btn1(color_conv2_1, 3, 32, 'conv2_2', is_training = is_training)
    print 022,color_conv2_2
    color_pool2   = layers.maxpool(color_conv2_2, 4,   'pool2')
    print 023, color_pool2
    # Block 3
    color_conv3_1 = layers.conv_btn1(color_pool2,   3, 64, 'conv3_1', is_training = is_training)
    print 031,color_conv3_1
    color_conv3_2 = layers.conv_btn1(color_conv3_1, 3, 64, 'conv3_2', is_training = is_training)
    print 032, color_conv3_2
    color_pool3   = layers.maxpool(color_conv3_2, 4,   'pool3')
    print 033, color_pool3
    color_drop3   = layers.dropout(color_pool3, dropout_keep_prob, 'drop3')
    print 034,color_drop3
    # Block 4
    color_conv4_1 = layers.conv_btn1(color_drop3,   3, 64, 'conv4_1', is_training = is_training)
    print 041,color_conv4_1
    color_conv4_2 = layers.conv_btn1(color_conv4_1, 3, 64, 'conv4_2', is_training = is_training)
    print 042,color_conv4_2
    color_pool4   = layers.maxpool(color_conv4_2, 4,   'pool4')
    color_drop4   = layers.dropout(color_pool4, dropout_keep_prob, 'drop4')
    print 044,color_drop4

    # Block 5
    color_conv5_1 = layers.conv_btn1(color_drop4,   3, 128, 'conv5_1', is_training = is_training)
    print 051,color_conv5_1
    color_conv5_2 = layers.conv_btn1(color_conv5_1, 3, 128, 'conv5_2', is_training = is_training)
    print 052,color_conv5_2
    color_drop5   = layers.dropout(color_conv5_2, dropout_keep_prob, 'drop5')
    print 055,color_drop5

    # Decoder Section
    # Block 1

    upsample61     = layers.deconv_upsample(color_drop5, 4,  'upsample6')
   # upsample61 = Cropping1D(cropping=((0, 1)))(upsample61)
    print "upsample61",upsample61
    concat6       = layers.concat(upsample61, color_conv4_2, 'concat6')
    print "concat6",concat6
    color_conv6_1 = layers.conv_btn1(concat6,       3, 128, 'conv6_1', is_training = is_training)
   # color_conv6_2 = layers.conv_btn1(color_conv6_1, 6, 128, 'conv6_2', is_training = is_training)
    color_drop6   = layers.dropout(color_conv6_1, dropout_keep_prob, 'drop6')
    print "color_drop6", color_drop6
    # Block 2
    upsample7     = layers.deconv_upsample(color_drop6, 4,  'upsample7')
    upsample7 = Cropping1D(cropping=((0, 1)))(upsample7)
    concat7       = layers.concat(upsample7, color_conv3_2, 'concat7')
    color_conv7_1 = layers.conv_btn1(concat7,       3, 64, 'conv7_1', is_training = is_training)
    print "color_conv7_1",color_conv7_1
   # color_conv7_2 = layers.conv_btn1(color_conv7_1, 6, 64, 'conv7_1', is_training = is_training)
    color_drop7   = layers.dropout(color_conv7_1, dropout_keep_prob, 'drop7')
    print "color_drop7",color_drop7

    # Block 3
    upsample81     = layers.deconv_upsample(color_drop7, 4,  'upsample8')
    upsample81 = Cropping1D(cropping=((1, 2)))(upsample81)
    concat8       = layers.concat(upsample81, color_conv2_2, 'concat8')
    color_conv8_1 = layers.conv_btn1(concat8,       3, 32, 'conv8_1', is_training = is_training)
    print "color_conv8_1",color_conv8_1
   # color_conv8_2 = layers.conv_btn1(color_conv8_1, 3, 32, 'conv8_1', is_training = is_training)

    # Block 4
    upsample91     = layers.deconv_upsample(color_conv8_1, 4, 'upsample9')
    upsample91 = Cropping1D(cropping=((1, 2)))(upsample91)
    concat9       = layers.concat(upsample91, color_conv1_2,  'concat9')
    color_conv9_1 = layers.conv_btn1(concat9,       3, 32,   'conv9_1', is_training = is_training)
    print "color_conv9_1", color_conv9_1
   # color_conv9_2 = layers.conv_btn1(color_conv9_1, 3, 32,   'conv9_1', is_training = is_training)

    # Block 5
    score  = layers.conv(color_conv9_1, 1, num_classes, 'score', activation_fn = None)
    print "score",score.shape
    logits = tf.reshape(score, (-1, num_classes))
    return logits

def segmentation_loss(logits, labels, class_weights = None):
    """
    Segmentation loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size * height * width, num_classes]
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None

    Returns:
        segment_loss: Segmentation loss
    """
    #import numpy as np
    print "33333333logits,labels", logits.shape, labels.shape
    labels = tf.to_int64(labels)
    label = tf.reshape(labels, [-1,3])
    #print "label_reshape",label
    #label = tf.expand_dims(label, -1)
    #print "label_expand",label
    #label_tile = tf.tile(label, (1, 3))
    #print "label_tile",label_tile
    #label_tile[:, 0] = tf.where(label_tile[:, 0] == 0,1,0)
    #label_tile[:, 1] = tf.where(label_tile[:, 1] == 1,1,0)
    #label_tile[:, 2] = tf.where(label_tile[:, 2] == 2,1,0)
    #label=tf.stack([x, y, z], axis=2)
   # label = label_tile
    label=tf.argmax(label, 1)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                             labels = label, logits = logits, name = 'segment_cross_entropy_per_example')

    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    #                    labels = label, logits = logits, name = 'segment_cross_entropy_per_example')
    print "cross_entropy",cross_entropy
    if class_weights is not None:
        weights = tf.matmul(label, class_weights, a_is_sparse = True)
        weights = tf.reshape(weights, [-1])
        cross_entropy = tf.multiply(cross_entropy, weights)
    segment_loss  = tf.reduce_mean(cross_entropy, name = 'segment_cross_entropy')

    tf.summary.scalar("loss/segmentation", segment_loss)

    return segment_loss


def l2_loss():
    """
    L2 loss:
    -------
    Returns:
        l2_loss: L2 loss for all weights
    """
    
    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

    tf.summary.scalar("loss/weights", l2_loss)

    return l2_loss


def loss(logits, labels, weight_decay_factor, class_weights = None):
    """
    Total loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size, height, width, 1]
        weight_decay_factor: float, factor with which weights are decayed
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None

    Returns:
        total_loss: Segmentation + Classification losses + WeightDecayFactor * L2 loss
    """

    segment_loss = segmentation_loss(logits, labels)
    total_loss   = segment_loss + weight_decay_factor * l2_loss()
    tf.summary.scalar("loss/total", total_loss)
    return total_loss

def accuracy(logits, labels):
    labels = tf.to_int64(labels)
    labels = tf.reshape(labels, [-1, 3])
            # tf.argmax: Returns the index with the largest value across axes of a tensor
    predicted_annots = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])
    predicted_labels = tf.reshape(tf.argmax(labels, axis=1), [-1, 1])
    precision=tf.metrics.mean_per_class_accuracy(predicted_labels,predicted_annots,3)
    #precision, recall, f1 = score(predicted_annots, predicted_labels)
    #precision = score(predicted_annots, predicted_labels)
    return precision

def train(loss, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, global_step):
    """
    Train opetation:
    ----------
    Args:
        loss: loss to use for training
        learning_rate: Float, learning rate
        learning_rate_decay_steps: Int, amount of steps after which to reduce the learning rate
        learning_rate_decay_rate: Float, decay rate for learning rate

    Returns:
        train_op: Training operation
    """
    
    decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 
                            learning_rate_decay_steps, learning_rate_decay_rate, staircase = True)

    # execute update_ops to update batch_norm weights
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer   = tf.train.AdamOptimizer(decayed_learning_rate)
        train_op    = optimizer.minimize(loss, global_step = global_step)

    tf.summary.scalar("learning_rate", decayed_learning_rate)

    return train_op


def predict(logits, batch_size, image_size):
    """
    Prediction operation:
    ----------------
    Args:
        logits: Tensor, predicted    [batch_size * height * width, num_classes]
        batch_size: Int, batch size
        image_size: Int, image width/height
    
    Returns:
        predicted_images: Tensor, predicted images   [batch_size, image_size, image_size]
    """

    predicted_images = tf.reshape(tf.argmax(logits, axis = 1), [batch_size, image_size])

    return predicted_images
