import argparse

license = "Rotator_loc"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
                                 description='', epilog=license, add_help=False)

network_config = parser.add_argument_group('network setting (must be provided)')

network_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/Rotator/ro_loc/')
network_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp102')
network_config.add_argument('--model1_name', type=str, dest='model1_name', default='Model01')
network_config.add_argument('--model2_name', type=str, dest='model2_name', default='Model02')
network_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
network_config.add_argument('--batch_size', type=int, dest='batch_size', default=8)
network_config.add_argument('--numEpoch', type=int, dest='num_epoch', default=0)  # infinite loop
network_config.add_argument('--trial_serial', type=int, dest='trial_serial', default=1)
network_config.add_argument('--npy_name', type=str, dest='npy_name', default='trval.npy')
network_config.add_argument('--max_keep', type=int, dest='max_keep', default=20)  # only use training
network_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
network_config.add_argument('--cam_type', type=str, dest='cam_type', default='pos')

#parser.print_help()
config, unparsed = parser.parse_known_args()

import sys, os
sys.path.append('/workspace/PycharmProjects/Rotator_loc')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore SSE instruction warning on tensorflow

import tensorflow as tf
import numpy as np
import sklearn.metrics  # roc curve
import matplotlib.pyplot as plt
import pandas as pd
import json, pptx, re
from pptx.util import Inches

trial_serial_str = '%03d' % (config.trial_serial)
log_path = os.path.join(config.data_path, config.exp_name, config.model1_name, 'logs-%s' % (trial_serial_str))
result_path = os.path.join(config.data_path, config.exp_name, config.model1_name, 'result-%s' % (trial_serial_str))
ckpt_path = os.path.join(result_path, 'ckpt')
npy_path = os.path.join(config.data_path, config.exp_name, 'npy')

if not os.path.exists(result_path): os.makedirs(result_path)
cam_path = os.path.join(config.data_path, 'cam')
if not os.path.exists(cam_path): os.makedirs(cam_path)
ppt_path = os.path.join(config.data_path, 'pptx')
if not os.path.exists(ppt_path): os.makedirs(ppt_path)

from data.tf_data import DataSetting, ImageProcess
import models.model_a as model_a
from tf_utils.tensor_board import TensorBoard
print('/'.join([npy_path, config.npy_name]))
data_set = DataSetting(data_dir=os.path.join(npy_path, config.npy_name), batch_size=config.batch_size,
                       only_val=bool(1 - config.train))

infer1_name = 'Inference' + config.model1_name
infer2_name = 'Inference' + config.model2_name
model1 = getattr(model_a, infer1_name)()
model2 = getattr(model_a, infer2_name)(trainable=config.train)


sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)


def training():
    ts_board = TensorBoard(log_dir=log_path, overwrite=True)
    mse = tf.get_variable(name='MSE', shape=[], trainable=False,
                          initializer=tf.zeros_initializer(), collections=['scalar'])
    loss = tf.get_variable(name='Entropy', shape=[], trainable=False,
                           initializer=tf.zeros_initializer(), collections=['scalar'])
    auc_rec = tf.get_variable(name='AUC', shape=[], trainable=False, initializer=tf.zeros_initializer(),
                              collections=['scalar'])
    accuracy_rec = tf.get_variable(name='Accuracy', shape=[], trainable=False, initializer=tf.zeros_initializer(),
                                   collections=['scalar'])

    ts_board.init_scalar(collections=['scalar'])
    ts_board.init_images(collections=['images'], num_outputs=4)

    train_op = model_a.optimize(model1.mse, model2.loss)

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver(max_to_keep=config.max_keep)

    result_name = '_'.join([config.exp_name, config.model1_name, trial_serial_str])+'.csv'
    auc_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'AUC': pd.Series()})

    crt_step, crt_epoch = None, None
    perf_per_epoch, max_perf_per_epoch, max_crt_step = [], [], []

    try:
        while True:
            sess.run([data_set.train.init_op, data_set.val.init_op])
            train_loss_batch, train_mse_batch, train_acc_batch = [], [], []
            train_x, train_y = [], []

            train_length = data_set.train.data_length
            num_iter_train = int(np.ceil(float(train_length) / config.batch_size))
            train_step = 0

            feed_dict = {}
            while train_step < num_iter_train:
                images1, xy_rp1, files1, labels, names = sess.run(data_set.train.next_batch)

                feed_local = {model1.images: images1, model1.locals: xy_rp1, model1.is_training: True}
                train_mse, train_loc = sess.run([model1.mse, model1.predict], feed_dict=feed_local)

                train_process = ImageProcess(files=files1, locals=train_loc, augmentation=True)

                feed_dict = {model2.images: train_process.crops, model2.labels: labels, model2.is_training: True,
                             model1.images: images1, model1.locals: xy_rp1, model1.is_training: True}

                _, train_loss, train_prob, train_acc, crt_step, crt_epoch = \
                    sess.run([train_op, model2.loss, model2.prob, model2.accuracy,
                              tf.train.get_global_step(), model2.global_epoch], feed_dict=feed_dict)

                sys.stdout.write('Step: {0:>4d} ({1})\r'.format(crt_step, int(crt_epoch)))
                
                train_x.extend(train_prob)
                train_y.extend(labels)
                train_acc_batch.append(train_acc)
                train_mse_batch.append(train_mse)
                train_loss_batch.append(train_loss)

                train_step += 1

            sess.run(tf.assign_add(model2.global_epoch, 1))
            
            fpr, tpr, _ = sklearn.metrics.roc_curve(train_y, train_x, drop_intermediate=False)
            train_auc = sklearn.metrics.auc(fpr, tpr)

            feed_dict.update({loss: np.mean(train_loss_batch), mse: np.mean(train_mse_batch),
                              auc_rec: train_auc, accuracy_rec: np.mean(train_acc_batch)})

            ts_board.add_summary(sess=sess, feed_dict=feed_dict, log_type='train')
            
            val_length = data_set.val.data_length
            num_iter_val = int(np.ceil(float(val_length) / config.batch_size))
            val_step = 0

            val_loss_batch, val_mse_batch, val_acc_batch = [], [], []
            val_x, val_y = [], []

            while val_step < num_iter_val:
                sys.stdout.write('Evaluation [{0}/{1}]\r'.format(len(val_loss_batch), num_iter_val))
                images1, xy_rp1, files1, labels, names = sess.run(data_set.val.next_batch)

                feed_local = {model1.images: images1, model1.locals: xy_rp1, model1.is_training: False}
                val_mse, val_loc = sess.run([model1.mse, model1.predict], feed_dict=feed_local)

                val_process = ImageProcess(files=files1, locals=val_loc, augmentation=False)

                feed_dict = {model2.images: val_process.crops, model2.labels: labels, model2.is_training: False,
                             model1.images: images1, model1.locals: xy_rp1, model1.is_training: False}

                val_loss, val_prob, val_acc = sess.run([model2.loss, model2.prob, model2.accuracy],
                                                       feed_dict=feed_dict)

                val_x.extend(val_prob)
                val_y.extend(labels)
                val_acc_batch.append(val_acc)
                val_mse_batch.append(val_mse)
                val_loss_batch.append(val_loss)

                val_step += 1
                
            fpr, tpr, _ = sklearn.metrics.roc_curve(val_y, val_x, drop_intermediate=False)
            val_auc = sklearn.metrics.auc(fpr, tpr)

            feed_dict.update({loss: np.mean(val_loss_batch), mse: np.mean(val_mse_batch),
                              auc_rec: val_auc, accuracy_rec: np.mean(val_acc_batch)})

            ts_board.add_summary(sess=sess, feed_dict=feed_dict, log_type='val')
            ts_board.display_summary(time_stamp=True)
            
            crt_epoch += 1
            if crt_epoch % 1 == 0:
                perf_per_epoch.append(val_auc)

                if crt_epoch < config.max_keep + 1:
                    max_crt_step.append(crt_step)
                    max_perf_per_epoch.append(val_auc)

                    saver.save(sess=sess, save_path=os.path.join(log_path, 'model.ckpt'),
                               global_step=crt_step)
                    auc_csv.loc[crt_step, 'WEIGHT_PATH'] = \
                        os.path.join(log_path, 'model.ckpt-'+str(crt_step))
                    auc_csv.loc[crt_step, 'AUC'] = val_auc

                elif val_auc > min(auc_csv['AUC'].tolist()):
                    auc_csv = auc_csv.drop(max_crt_step[0])
                    max_crt_step.pop(0)
                    max_crt_step.append(crt_step)
                    max_perf_per_epoch.pop(0)
                    max_perf_per_epoch.append(val_auc)

                    saver.save(sess=sess, save_path=os.path.join(log_path, 'model.ckpt'),
                               global_step=crt_step)
                    auc_csv.loc[crt_step, 'WEIGHT_PATH'] = \
                        os.path.join(log_path, 'model.ckpt-'+str(crt_step))
                    auc_csv.loc[crt_step, 'AUC'] = val_auc

                auc_csv.to_csv(os.path.join(result_path, result_name))

                if crt_epoch == 300: break
        print('Training Complete...\n')
        sess.close()

    except KeyboardInterrupt:
        print('Result saved')
        auc_csv.to_csv(os.path.join(result_path, result_name))


def validation():
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    saver = tf.train.Saver()
    num_examples = data_set.val.data_length

    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model1_name,
                                   'result-%03d' % config.trial_serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path,
                                              '_'.join([config.exp_name, config.model1_name,
                                                        '%03d' % config.trial_serial])+'.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('AUC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])

    num_ckpt = len(all_ckpt_paths)
    print('num_ckpt: ', num_ckpt)

    imgs = np.zeros([num_examples, model2.img_h, model2.img_w, model2.img_c])
    cams = np.zeros([num_ckpt, num_examples, model2.img_h, model2.img_w, model2.img_c])

    lbls = np.zeros([num_examples, ], dtype=np.int32)
    probs = np.zeros([num_ckpt, num_examples, 1])

    val_x, val_y = None, None
    for ckpt_idx, ckpt_path in enumerate(all_ckpt_paths):
        w_path = os.path.basename(ckpt_path)
        ckpt_path = os.path.join(log_path, w_path)
        print('Restoring: '+ckpt_path)

        sess = tf.Session(config=sess_config)
        saver.restore(sess, ckpt_path)

        sess.run(data_set.val.init_op)

        val_x, val_y = [], []
        num_iter = int(np.ceil(float(num_examples) / config.batch_size))
        step = 0

        while step < num_iter:
            images1, xy_rp1, files1, labels, names = sess.run(data_set.val.next_batch)

            feed_local = {model1.images: images1, model1.locals: xy_rp1, model1.is_training: False}
            val_mse, val_loc = sess.run([model1.mse, model1.predict], feed_dict=feed_local)

            val_process = ImageProcess(files=files1, locals=val_loc, augmentation=False)

            feed_dict = {model2.images: val_process.crops, model2.labels: labels, model2.is_training: False,
                         model1.images: images1, model1.locals: xy_rp1, model1.is_training: False}

            val_prob, val_acc, val_cam = sess.run([model2.prob, model2.accuracy, model2.local_pos],
                                                  feed_dict=feed_dict)

            val_x.extend(val_prob)
            val_y.extend(labels)

            probs[ckpt_idx, step * config.batch_size:step * config.batch_size + len(labels)] = val_prob
            cams[ckpt_idx, step * config.batch_size:step * config.batch_size + len(labels)] = val_cam

            if ckpt_idx == 0:
                lbls[step * config.batch_size:step * config.batch_size + len(labels)] = labels
                imgs[step * config.batch_size:step * config.batch_size + len(labels)] = val_process.crops

            step += 1
        sess.close()

    probs, cams = np.mean(probs, axis=0), np.mean(cams, axis=0)
    prob_1 = np.squeeze(np.array(probs))

    result_csv = pd.DataFrame({'NUMBER': data_set.val.id, 'PROB': prob_1, 'LABEL': np.array(lbls)})
    result_name = '_'.join([config.model1_name, config.npy_name, trial_serial_str,
                            '%03d' % (config.num_weight)])+'.csv'
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)

    fpr, tpr, _ = sklearn.metrics.roc_curve(val_y, prob_1, drop_intermediate=False)
    val_auc = sklearn.metrics.auc(fpr, tpr)


    print('Validation AUC: ', val_auc)
    print('Validation Complete...\n')

    prs = pptx.Presentation()
    prs.slide_width, prs.slide_height = Inches(8*2), Inches(5*2)

    plt_batch = 20
    plt_step = 0
    plt_iter, plt_examples = int(np.ceil(num_examples / plt_batch)), num_examples

    while plt_step < plt_iter:
        if plt_examples >= plt_batch:
            len_batch = plt_batch
        else:
            len_batch = plt_examples

        labels_batch = lbls[plt_step * plt_batch:plt_step * plt_batch + len_batch]
        names_batch = data_set.val.id[plt_step * plt_batch:plt_step*plt_batch + len_batch]
        probs_batch = probs[plt_step * plt_batch:plt_step * plt_batch + len_batch]

        images_batch = imgs[plt_step * plt_batch:plt_step*plt_batch + len_batch]
        cams_batch = cams[plt_step * plt_batch:plt_step * plt_batch + len_batch]

        blank_slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_slide_layout)

        show_cam(cams_batch, probs_batch, images_batch, labels_batch, names_batch, 'LABEL')
        fig_name = '_'.join([config.exp_name, config.model1_name, config.npy_name, trial_serial_str])
        fig_path = os.path.join(cam_path, fig_name)+'.png'
        plt.savefig(fig_path, bbox_inches='tight')
        slide.shapes.add_picture(fig_path, Inches(0), Inches(0), width=Inches(8*2))
        os.remove(fig_path)
        plt_step += 1
        plt_examples -= plt_batch

    print('plt_examples check: ', plt_examples)
    ppt_name = os.path.join(ppt_path, '_'.join([config.exp_name, config.model1_name, config.npy_name,
                                                trial_serial_str, '%03d' % config.num_weight])+'.pptx')
    prs.save(ppt_name)
    print('Saved: ', ppt_name)

    
def show_cam(cams, probs, images, labels, names, side_label, num_rows=5, num_cols=8, figsize=(8*2, 5*2)):
    batch_size = cams.shape[0]
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
    axoff_fun(ax)

    for i in range(batch_size):
        prob, lbl = '%.2f' % probs[i], int(labels[i])
        show_image, cam = np.squeeze(images[i]), np.squeeze(cams[i])
        img_row, img_col = int(i % num_rows), int(i / num_rows) * 2

        ori_title = ' '.join([names[i], side_label, ': '+str(lbl)])
        cam_title = side_label+' Pred: '+str(prob)

        ax[img_row, img_col].imshow(show_image, cmap='bone')
        ax[img_row, img_col+1].imshow(show_image, cmap='bone')
        ax[img_row, img_col+1].imshow(cam, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')

        if (lbl == 0 and probs[i] < 0.5) or (lbl == 1 and probs[i] >= 0.5):
            txt_color = 'blue'
        else:
            txt_color = 'red'
        ax[img_row, img_col].set_title(ori_title, fontsize=7, color=txt_color)
        ax[img_row, img_col+1].set_title(cam_title, fontsize=7, color=txt_color)


if __name__ == '__main__':
    if config.train:
        training()
    else:
        validation()
