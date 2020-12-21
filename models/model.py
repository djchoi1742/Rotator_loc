import tensorflow as tf
import numpy as np
import sys, re
import os, json
sys.path.append('/home/chzze/bitbucket/Rotator_loc')


from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


def get_info(data_path, key):
    with open(os.path.join(data_path, '.info'), 'r') as f:
        info = json.load(f)
        return info[key]


def design_scope(class_name, output_name='Classifier'):
    model_scope = re.sub('Inference', '', class_name)
    classifier_scope = re.sub('Model', output_name, model_scope)
    return model_scope, classifier_scope


def calculate_accuracy(prob, label):
    predicted = tf.cast(prob > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label), dtype=tf.float32))
    return accuracy


def bottleneck_layer(input_x, growth_k, is_training, use_dropout=False, dropout_rate=0.2):
    out = layers.batch_norm(inputs=input_x, center=True, scale=True, is_training=is_training)
    out = tf.nn.relu(out)
    out = layers.conv2d(inputs=out, num_outputs=4*growth_k, kernel_size=1, stride=1,
                        weights_initializer=layers.variance_scaling_initializer(),
                        padding='SAME', activation_fn=None)

    out = layers.batch_norm(inputs=out, center=True, scale=True, is_training=is_training)
    out = tf.nn.relu(out)
    out = layers.conv2d(inputs=out, num_outputs=growth_k, kernel_size=3, stride=1,
                        weights_initializer=layers.variance_scaling_initializer(),
                        padding='SAME', activation_fn=None)
    if use_dropout:
        out = tf.layers.dropout(inputs=out, rate=dropout_rate, training=is_training)
    return out


def transition_layer(input_x, layer_name, is_training, theta=1.0, reduction_ratio=16, last_layer=False):
    with tf.name_scope(layer_name):
        in_channel = input_x.shape[-1].value
        out = layers.batch_norm(inputs=input_x, center=True, scale=True, is_training=is_training)
        out = tf.nn.relu(out)
        out = layers.conv2d(inputs=out, num_outputs=int(in_channel*theta), kernel_size=1, stride=1,
                            weights_initializer=layers.variance_scaling_initializer(),
                            padding='SAME', activation_fn=None)

        if last_layer is False:
            squeeze = tf.reduce_mean(out, axis=[1, 2], keepdims=True)  # global average pooling
            excitation = layers.fully_connected(inputs=squeeze,
                                                num_outputs=squeeze.shape[-1].value // reduction_ratio,
                                                weights_initializer=layers.variance_scaling_initializer(), # Add
                                                activation_fn=tf.nn.relu)
            excitation = layers.fully_connected(inputs=excitation,
                                                num_outputs=squeeze.shape[-1].value,
                                                weights_initializer=layers.variance_scaling_initializer(), # Add
                                                activation_fn=tf.nn.sigmoid)
            se_out = out * excitation
            avg_pool = layers.avg_pool2d(inputs=se_out, kernel_size=[2, 2], stride=2, padding='SAME')

            print(avg_pool)
        else:
            avg_pool = out

    return avg_pool


def dense_block(input_x, layer_name, rep, growth_k, is_training, use_dropout=False, use_se=False,
                reduction_ratio=16):

    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, growth_k, is_training, use_dropout)
        layers_concat.append(x)

        for i in range(rep - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer(x, growth_k, is_training, use_dropout)
            layers_concat.append(x)
        x = tf.concat(layers_concat, axis=3)

        if use_se:
            squeeze = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # global average pooling
            excitation = layers.fully_connected(inputs=squeeze,
                                                num_outputs=squeeze.shape[-1].value // reduction_ratio,
                                                weights_initializer=layers.variance_scaling_initializer(),  # Add
                                                activation_fn=tf.nn.relu)
            excitation = layers.fully_connected(inputs=excitation,
                                                num_outputs=squeeze.shape[-1].value,
                                                weights_initializer=layers.variance_scaling_initializer(),  # Add
                                                activation_fn=tf.nn.sigmoid)
            x = x * excitation
    print(x)
    return x


def focal_loss_sigmoid(labels, logits, alpha=0.25, gamma=2):
    """
    Compute focal loss for binary classification
    Args:
    :param labels: A int32 tensor of shape [batch_size].
    :param logits: A float 32 tensor of shape [batch_size].
    :param alpha: A scalar for focal loss alpha hyper-parameter. If positive sample number
                  > negative sample number, alpha < 0.5 and vice versa.
    :param gamma: A scalar for focal loss gamma hyper-parameter.
    :return: A tensor of the same shape as 'labels'
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    fcl_loss = -labels*(1-alpha)*((1-y_pred)**gamma)*tf.log(y_pred)-(1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return fcl_loss


class InferenceModel03:
    def __init__(self, image_size=256, image_height=256, growth_k=32, theta=0.5,
                 block_rep='2,2,2,2,2', k_p='1,1,1,1,1', use_se=False, view=1, **kwargs):

        # self.model_scope, self.localize_scope = design_scope(type(self).__name__, 'Localize')
        model_scope, localize_scope = design_scope(type(self).__name__, 'Localize')
        self.model_scope = '_'.join(['v'+str(view), model_scope])
        self.localize_scope = '_'.join(['v'+str(view), localize_scope])

        self.img_h, self.img_w, self.img_c = image_height, image_size, 1
        self.local_num = 2
        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.locals = tf.placeholder(tf.float32, shape=[None, self.local_num], name='Localization')
        self.is_training = tf.placeholder(tf.bool, shape=None)
        self.growth_k = growth_k

        block_rep_list = list(map(int, re.split(',', block_rep)))
        block_num = len(block_rep_list)
        k_p_list = list(map(int, re.split(',', k_p)))

        with tf.variable_scope(self.model_scope):
            features_2d = self.images
            first_conv = layers.conv2d(inputs=features_2d, num_outputs=2 * self.growth_k, kernel_size=[7, 7],
                                       weights_initializer=layers.variance_scaling_initializer(), # Add
                                       stride=2, activation_fn=None)
            print('1st conv: ', first_conv)

            first_pool = layers.max_pool2d(inputs=first_conv, kernel_size=[3, 3], stride=2, padding='SAME')
            print('1st pool: ', first_pool)

            dsb = first_pool

            for i in range(0, block_num-1):
                dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                                  growth_k=k_p_list[i]*growth_k, use_se=use_se, is_training=self.is_training)

                dsb = transition_layer(input_x=dsb, layer_name='Transition'+str(i+1),
                                       theta=theta, is_training=self.is_training)

            self.last_dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                                        rep=block_rep_list[-1], growth_k=k_p_list[-1]*growth_k,
                                        use_se=use_se, is_training=self.is_training)

            self.bn_relu = tf.nn.relu(layers.batch_norm(self.last_dsb,
                                                        center=True, scale=True, is_training=self.is_training))

            self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)

        with tf.variable_scope(self.localize_scope):

            flatten = layers.flatten(self.last_pool)
            self.fc = layers.fully_connected(inputs=flatten, num_outputs=flatten.shape[-1].value,
                                             weights_initializer=layers.variance_scaling_initializer(), # Add
                                             activation_fn=tf.nn.relu)
            print('fc: ', self.fc)
            self.reg_fc = layers.fully_connected(inputs=self.fc, num_outputs=self.local_num,
                                                 weights_initializer=layers.variance_scaling_initializer(), # Add
                                                 activation_fn=None)
            print('reg_fc: ', self.reg_fc)
            self.residual = tf.square(self.reg_fc - self.locals)

            self.mse = tf.multiply(tf.reduce_mean(tf.square(self.reg_fc - self.locals)), 1.0)
            print('mse: ', self.mse)
            self.predict = tf.multiply(self.reg_fc, image_height)
            print('predict: ', self.predict)


def training_option(loss, learning_rate=0.01, decay_steps=5000, decay_rate=0.94, decay=0.9, epsilon=0.1):
    with tf.variable_scope('reg_loss'):
        reg_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
    lr_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=tf.train.get_global_step(),
                                         decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=decay, epsilon=epsilon)
        train = optimizer.minimize(loss+reg_loss, global_step=tf.train.get_global_step())
    return train


def optimize(ce_loss, reg_loss, learning_rate=0.01, decay_steps=5000, decay_rate=0.94, decay=0.9, epsilon=0.1):
    with tf.name_scope('reg_loss'):
        r2_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

    with tf.name_scope('Total_loss'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, epsilon=epsilon)
            train = optimizer.minimize(ce_loss + reg_loss + r2_loss, global_step=tf.train.get_global_step())
    return train


class InferenceModel04:
    def __init__(self, trainable=False, image_size=512, image_height=512, growth_k=32, theta=0.5,
                 alpha=0.3, gamma=2, block_rep='3,3,3,3', k_p='1,1,1,1', use_se=False, view=1, **kwargs):

        # self.model_scope, self.classify_scope = design_scope(type(self).__name__, 'Classify')
        model_scope, classify_scope = design_scope(type(self).__name__, 'Classify')
        self.model_scope = '_'.join(['v'+str(view), model_scope])
        self.classify_scope = '_'.join(['v'+str(view), classify_scope])

        self.img_h, self.img_w, self.img_c = image_height, image_size, 1
        self.class_num = 1
        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.labels = tf.placeholder(tf.float32, shape=[None], name='Raw_label')
        self.is_training = tf.placeholder(tf.bool, shape=None, name='Classify_train')
        self.growth_k = growth_k

        tf.add_to_collection('crop_images', tf.boolean_mask(self.images, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('crop_images', tf.boolean_mask(self.images, tf.equal(self.labels, 1), name='positive'))

        block_rep_list = list(map(int, re.split(',', block_rep)))
        block_num = len(block_rep_list)
        k_p_list = list(map(int, re.split(',', k_p)))

        with tf.variable_scope(self.model_scope):
            features_2d = self.images
            first_conv = layers.conv2d(inputs=features_2d, num_outputs=2 * self.growth_k, kernel_size=[7, 7],
                                       weights_initializer=layers.variance_scaling_initializer(), # Add
                                       stride=2, activation_fn=None)
            print('1st conv: ', first_conv)

            first_pool = layers.max_pool2d(inputs=first_conv, kernel_size=[3, 3], stride=2, padding='SAME')
            print('1st pool: ', first_pool)

            dsb = first_pool

            for i in range(0, block_num-1):
                dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                                  growth_k=k_p_list[i]*growth_k, use_se=use_se, is_training=self.is_training)
                dsb = transition_layer(input_x=dsb, layer_name='Transition'+str(i+1),
                                       theta=theta, is_training=self.is_training)

            self.last_dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                                        rep=block_rep_list[-1], growth_k=k_p_list[-1]*growth_k,
                                        use_se=use_se, is_training=self.is_training)

            self.bn_relu = tf.nn.relu(layers.batch_norm(self.last_dsb,
                                                        center=True, scale=True, is_training=self.is_training))

            self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)

        with tf.variable_scope(self.classify_scope):
            flatten = layers.flatten(self.last_pool)
            print('flatten: ', flatten)
            self.fc = layers.fully_connected(inputs=flatten, num_outputs=flatten.shape[-1].value,
                                             weights_initializer=layers.variance_scaling_initializer(), # Add
                                             activation_fn=tf.nn.relu)
            print('fc: ', self.fc)
            self.logits = layers.fully_connected(inputs=self.fc, num_outputs=self.class_num,
                                                 weights_initializer=layers.variance_scaling_initializer(), # Add
                                                 activation_fn=None)

            self.prob = tf.nn.sigmoid(self.logits)
            print('prob: ', self.prob)

            labels = tf.expand_dims(self.labels, axis=-1)
            self.accuracy = calculate_accuracy(self.prob[:, 0], tf.cast(self.labels, dtype=tf.float32))
            focal_loss = focal_loss_sigmoid(labels=labels, logits=self.logits, alpha=alpha, gamma=gamma)
            self.loss = tf.reduce_mean(focal_loss)

            grad_con = tf.reduce_mean(tf.gradients(self.prob, self.bn_relu)[0], axis=[1, 2], keepdims=True)
            self.local1 = tf.reduce_mean(grad_con * self.bn_relu, axis=-1, keepdims=True)
            self.local = tf.image.resize_bilinear(images=self.local1, size=[self.img_h, self.img_w])
            print('local: ', self.local)

        if trainable:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
            
            
def each_param(train_var):
    var_shape = train_var.shape.as_list()
    var_param = np.prod(np.array(var_shape))
    return var_param


if __name__ == '__main__':
    InferenceModel03(block_rep='2,2,2,2,2', view=1)
    InferenceModel04(block_rep='3,3,3,3', view=1)
    train_vars = tf.trainable_variables()
    total_params = list(map(each_param, train_vars))

    print(train_vars)
    print(len(train_vars))
    print(sum(total_params))





            
