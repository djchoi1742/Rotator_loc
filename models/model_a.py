import tensorflow as tf
import numpy as np
import sys, re
sys.path.append('/workspace/PycharmProjects/Rotator_loc')

from tensorflow.contrib import layers
from data.tf_data import IMAGE_SIZE1, IMAGE_SIZE2


def design_scope(class_name):
    model_scope = re.sub('Inference', '', class_name)
    classifier_scope = re.sub('Model', 'Classifier', model_scope)
    return model_scope, classifier_scope


def se_res_block(inputs, in_channels, out_channels, use_attention, last_block, is_training):
    conv = layers.conv2d(inputs=inputs, num_outputs=in_channels, kernel_size=1, stride=1, activation_fn=None)
    conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

    conv = layers.conv2d(inputs=conv, num_outputs=in_channels, kernel_size=3, stride=1, activation_fn=None)
    conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

    conv = layers.conv2d(inputs=conv, num_outputs=out_channels, kernel_size=1, stride=1, activation_fn=None)
    conv = layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training)

    def attention_block(inputs, reduction_ratio=4):
        squeeze = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)  # global average pooling
        excitation = layers.fully_connected(inputs=squeeze,
                                            num_outputs=squeeze.shape[-1].value // reduction_ratio,
                                            activation_fn=tf.nn.relu)
        excitation = layers.fully_connected(inputs=excitation, num_outputs=squeeze.shape[-1].value,
                                            activation_fn=tf.nn.sigmoid)
        outputs = inputs * excitation
        return outputs

    if use_attention:
        conv = attention_block(inputs=conv, reduction_ratio=16)

    if not inputs.shape[-1].value == out_channels:
        inputs = layers.conv2d(inputs=inputs, num_outputs=out_channels,
                               kernel_size=1, stride=1, activation_fn=None)
        inputs = layers.batch_norm(inputs=inputs, center=True, scale=True, is_training=is_training)

    if last_block:
        return conv + inputs
    else:
        return tf.nn.relu(conv + inputs)


def calculate_accuracy(prob, label):
    predicted = tf.cast(prob > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label), dtype=tf.float32))
    return accuracy


class InferenceModel01:
    def __init__(self):
        self.model_scope, self.classifier_scope = design_scope(class_name = type(self).__name__)
        image_height, image_width = IMAGE_SIZE1[0], IMAGE_SIZE1[1]
        self.img_h, self.img_w, self.img_c = [image_height, image_width, 1]
        self.local_num = 2
        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c],
                                     name='Raw_image')
        self.locals = tf.placeholder(tf.float32, shape=[None, self.local_num], name='Localization')
        self.is_training = tf.placeholder(tf.bool, shape=None, name='Localize_train')

        tf.add_to_collection('images', self.images)

        features_2d = self.images
        with tf.variable_scope(self.model_scope):
            for in_channels, out_channels in zip([64,96,128,160,192], [128,160,192,224,256]):
                features_2d = se_res_block(inputs=features_2d,
                                           in_channels=in_channels, out_channels=out_channels,
                                           use_attention=True, last_block=False,
                                           is_training=self.is_training)
                features_2d = layers.max_pool2d(inputs=features_2d, kernel_size=2, stride=2)
                print(features_2d)

        with tf.variable_scope(self.classifier_scope):
            self.last_conv = layers.conv2d(inputs=features_2d, num_outputs=image_width, kernel_size=3,
                                             stride=1, activation_fn=None)
            self.features_2d = tf.nn.relu(layers.batch_norm(inputs=self.last_conv, center=True, scale=True,
                                                            is_training=self.is_training))
            k_size = int(self.features_2d.shape[1])
            avg_pool = layers.avg_pool2d(inputs=self.features_2d, kernel_size=k_size, stride=k_size)
            flatten = tf.reshape(avg_pool, [-1, int(avg_pool.shape[-1])])

        self.reg_fc = layers.fully_connected(inputs=flatten, num_outputs=self.local_num, activation_fn=None)
        self.mse = tf.multiply(tf.reduce_mean(tf.square(self.reg_fc - self.locals)), 1.0)
        self.predict = tf.multiply(self.reg_fc, image_height)

        
def optimize(ce_loss, reg_loss, learning_rate=0.01, decay_steps=5000, decay_rate=0.94, decay=0.9, epsilon=0.1):
    # regularization loss
    with tf.name_scope('reg_loss'):
        r2_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, epsilon=epsilon)
        train = optimizer.minimize(ce_loss + reg_loss + r2_loss, global_step=tf.train.get_global_step())
    return train


class InferenceModel02:
    def __init__(self, trainable=False):
        self.model_scope, self.classifier_scope = design_scope(class_name = type(self).__name__)
        image_height, image_width = IMAGE_SIZE2[0], IMAGE_SIZE2[1]
        self.img_h, self.img_w, self.img_c = [image_height, image_width, 1]
        self.class_num = 2
        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c],
                                     name='Crop_image')
        self.labels = tf.placeholder(tf.int64, shape=[None], name='Raw_label')  # sparse index
        self.is_training = tf.placeholder(tf.bool, shape=None, name='Classify_train')

        tf.add_to_collection('crop_images', tf.boolean_mask(self.images, tf.equal(self.labels, 0),
                                                            name='negative'))
        tf.add_to_collection('crop_images', tf.boolean_mask(self.images, tf.equal(self.labels, 1),
                                                            name='positive'))

        features_2d = self.images
        with tf.variable_scope(self.model_scope):
            for in_channels, out_channels in zip([64,96,128,160,192], [128,160,192,224,256]):
                features_2d = se_res_block(inputs=features_2d,
                                           in_channels=in_channels, out_channels=out_channels,
                                           use_attention=True, last_block=False,
                                           is_training=self.is_training)
                features_2d = layers.max_pool2d(inputs=features_2d, kernel_size=2, stride=2)
                print(features_2d)

        with tf.variable_scope(self.classifier_scope):
            for in_channels, out_channels in zip([224], [self.class_num]):
                self.features_2d = se_res_block(inputs=features_2d,
                                                in_channels=in_channels, out_channels=out_channels,
                                                use_attention=True, last_block=True,
                                                is_training=self.is_training)
            print('last conv: ', self.features_2d)

            # aggregation of local classification into the global classification result
            self.logits = tf.reduce_logsumexp(self.features_2d, axis=[1,2], keepdims=False)

        self.prob = tf.expand_dims(tf.nn.softmax(self.logits)[:,1], -1)

        is_correct = tf.equal(tf.argmax(self.logits, axis=1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('prob: ', self.prob)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=self.logits))

        self.local_class = tf.image.resize_bilinear(images=self.features_2d, size=[self.img_h, self.img_w])
        self.local_neg = tf.expand_dims(tf.nn.relu(self.local_class[:, :, :, 0]), axis=-1)
        self.local_pos = tf.expand_dims(tf.nn.relu(self.local_class[:, :, :, 1]), axis=-1)
        print('local_pos: ', self.local_pos)

        if trainable:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')


if __name__ == '__main__':
    InferenceModel01()
    print('\n')
    InferenceModel02()
