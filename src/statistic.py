import tensorflow as tf
import numpy as np
from utils import data_handler as dh
import src.OPTICS as OPTICS
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as aps
FLAGS = tf.app.flags.FLAGS
from time import time
import config
import os
import pickle
import csv


class Statistic():
    def __init__(self, logits, total_loss, loss_list, input_ph, ground_truths_ph, multi_loss_class, processed_ground_truths):
        self.labelsID_outpot = logits[0]
        self.InstanceID_outpot = logits[1]
        self.Disparity_outpot = logits[2]
        self.total_loss = total_loss
        self.loss_list = loss_list
        self.input_ph = input_ph
        self.ground_truths_ph = ground_truths_ph
        self.multi_loss_class = multi_loss_class
        self.processed_ground_truths = processed_ground_truths
        self.epoch_num = 0
        # lists for saving statistic over time
        self.eval_keys = ['total_loss', 'labelsID_acc', 'InstanceID_per_pixel_rms', 'InstanceID_total_rms',
                          'Label_ap_acc', 'Disparity_per_pixel_rms', 'Disparity_total_rms']
        self.statistics = dict((k, {'val': [], 'train': []}) for k in self.eval_keys)
        self.statistics['loss_lists'] = {'val': [[] for i in range(len(loss_list))], 'train': [[] for i in range(len(loss_list))]}
        self.statistics['sigmas_list'] = [[] for i in range(len(loss_list))]
        self.statistics['weights_list'] = [[] for i in range(len(loss_list))]
        self.statistics['label_scores'] = {}
        self.statistics['label_scores']['acc'] = {'val': [], 'train': []}
        self.statistics['label_scores']['cl_acc_mean'] = {'val': [], 'train': []}
        self.statistics['label_scores']['iu_mean'] = {'val': [], 'train': []}
        self.statistics['label_scores']['cl_acc'] = {'val': [], 'train': []}
        self.statistics['label_scores']['iu'] = {'val': [], 'train': []}
        self.statistics['label_scores']['iu_no_void_mean'] = {'val': [], 'train': []}
        # self.label_scores['label_ap_score'] = {'val': [], 'train': []}
        # self.label_scores['instance_ap_score'] = {'val': [], 'train': []}
        # make plot dirs
        self.make_dirs()

    def make_dirs(self):
        if not os.path.exists(FLAGS.stat_dump_path):
            os.makedirs(FLAGS.stat_dump_path)
        if not os.path.exists(FLAGS.stat_dump_path):
            os.makedirs(FLAGS.stat_csv_path)
        if not os.path.exists(FLAGS.plots_path):
            os.makedirs(FLAGS.plots_path)
        if not os.path.exists(FLAGS.example_preds):
            os.makedirs(FLAGS.example_preds)
        if not os.path.exists(os.path.join(FLAGS.example_preds, 'label')):
            os.makedirs(os.path.join(FLAGS.example_preds, 'label'))
        if not os.path.exists(os.path.join(FLAGS.example_preds, 'disp')):
            os.makedirs(os.path.join(FLAGS.example_preds, 'disp'))
        if not os.path.exists(os.path.join(FLAGS.example_preds, 'instance', "y_reg")):
            os.makedirs(os.path.join(FLAGS.example_preds, 'instance', "y_reg"))
        if not os.path.exists(os.path.join(FLAGS.example_preds, 'instance', "x_reg")):
            os.makedirs(os.path.join(FLAGS.example_preds, 'instance', "x_reg"))

    def arange_eval_lists(self, val_eval_dict, train_eval_dict):
        [self.statistics[key]['val'].append(val_eval_dict[key]) for key in self.eval_keys if
         key not in ['Label_ap_acc', 'loss_lists']]   # Those are updated differently
        [self.statistics[key]['train'].append(train_eval_dict[key]) for key in self.eval_keys if
         key not in ['Label_ap_acc', 'loss_lists']]
        if (self.epoch_num + 1) % FLAGS.calc_ap_epoch_num == 0:
            self.statistics['Label_ap_acc']['val'].append(val_eval_dict['Label_ap'])
            self.statistics['Label_ap_acc']['train'].append(train_eval_dict['Label_ap'])
        for loss_num in range(len(val_eval_dict['loss_list'])):
            self.statistics['loss_lists']['val'][loss_num].append(val_eval_dict['loss_list'][loss_num])
            self.statistics['loss_lists']['train'][loss_num].append(train_eval_dict['loss_list'][loss_num])

    def handle_statistic(self, epoch, logits, sess, train_input_imgs=None, train_gts=None,
                        val_input_imgs=None, val_gts=None, verbose=1):
        self.epoch_num = epoch
        if epoch % FLAGS.val_epoch == 0:
            start = time()
            val_eval_dict = self.run_evaluation('val', logits, sess, input_imgs=val_input_imgs, gts=val_gts)
            middle = time()
            print('time for val eval: %.2f' %(middle-start))
            train_eval_dict = self.run_evaluation('train', logits, sess, input_imgs=train_input_imgs, gts=train_gts)
            end = time()
            print('time for train eval: %.2f' % (end - middle))
            self.set_sigmas_and_wights()
            self.arange_eval_lists(val_eval_dict, train_eval_dict)
            self.save_plots()
            # self.save_cvss()
            if verbose:
                self.print_end_epoch_stats(train_eval_dict, val_eval_dict)

        if epoch % FLAGS.example_epoch == 0:
            start = time()
            self.calc_and_save_examples(logits, sess)
            end = time()
            print('calc example time: %.2f' % (end - start))

    # def save_csv(self):
    #     with open(os.path.join(FLAGS.stat_csv_path, 'stat.csv'), 'wb') as myfile:
    #         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #         wr.writerow(mylist)
#

    def print_end_epoch_stats(self, train_eval_dict, val_eval_dict):
        print('train - label accuracy: %.2f' % (train_eval_dict['labelsID_acc'])
              + ', label IoU: %.2f' % (self.statistics['label_scores']['iu_mean']['train'][-1])
              + ', best IoU: %.2f' % (np.max(self.statistics['label_scores']['iu_mean']['train'])))
        print('val   - label accuracy: %.2f' % (val_eval_dict['labelsID_acc'])
              + ', label IoU: %.2f' % (self.statistics['label_scores']['iu_mean']['val'][-1])
              + ', best IoU: %.2f' % (np.max(self.statistics['label_scores']['iu_mean']['val'])))
        print('train - Instance RMS per pixel: %.4f' % (train_eval_dict['InstanceID_per_pixel_rms'])
              + ', Instance total RMS: %.4f' % (train_eval_dict['InstanceID_total_rms']))
        print('val   - Instance RMS per pixel: %.4f' % (val_eval_dict['InstanceID_per_pixel_rms'])
              + ', Instance total RMS: %.4f' % (val_eval_dict['InstanceID_total_rms']))
        print('train - disp RMS per pixel: %.4f' % (train_eval_dict['Disparity_per_pixel_rms'])
              + ', disp total RMS: %.4f' % (train_eval_dict['Disparity_total_rms']))
        print('val   - disp RMS per pixel: %.4f' % (val_eval_dict['Disparity_per_pixel_rms'])
              + ', disp total RMS: %.4f' % (val_eval_dict['Disparity_total_rms']))

    def run_evaluation(self, set_name, logits, sess, input_imgs=None, gts=None):
        labelsID_acc_sum = 0
        InstanceID_per_pixel_rms_sum = 0
        InstanceID_total_rms_sum = 0
        disp_per_pixel_rms_sum = 0
        disp_total_rms_sum = 0
        total_loss_sum = 0
        Label_ap = 0    # average precision
        loss_list_sum = [0]*len(self.loss_list)
        n = len(config.colors[config.working_dataset])
        label_id_hist = np.zeros((n, n))  # n = 34 (num of classes)
        num_per_set = {'train': FLAGS.num_of_train_imgs,
                       'val': FLAGS.num_of_val_imgs}
        set_images_number = num_per_set[set_name]
        input_len = 0
        for ind in range(set_images_number):
            if input_imgs is not None:
                image, gt = input_imgs[ind], gts[ind]
            else:
                image, gt = dh.get_data(ind, set_name)
            processed_gts = [None] * 3
            if FLAGS.need_resize:
                processed_gts[0] = scipy.misc.imresize(gt[0].squeeze(), (FLAGS.output_height, FLAGS.output_width))
                processed_gts[1] = scipy.misc.imresize(gt[1].squeeze(), (FLAGS.output_height, FLAGS.output_width))
                processed_gts[2] = scipy.misc.imresize(gt[2].squeeze(), (FLAGS.output_height, FLAGS.output_width))
            else:
                processed_gts[0] = gt[0].squeeze()
                processed_gts[1] = gt[1].squeeze()
                processed_gts[2] = gt[2].squeeze()
            full_feed_dict = self.get_feed_dict(image, gt)
            run_list = self.get_run_list(logits)
            pred_list = sess.run(run_list, feed_dict=full_feed_dict)
            labelsID_acc_sum += self.calc_labelsID_acc(pred_list[0], pred_list[4])
            label_id_hist += self.fast_hist(pred_list[0], pred_list[4], n)
            per_pixel_rms, total_rms = self.calc_InstanceID_rms(pred_list[1], processed_gts[1])
            InstanceID_per_pixel_rms_sum += per_pixel_rms
            InstanceID_total_rms_sum += total_rms
            per_pixel_rms_disp, total_rms_disp = self.calc_Disparity_rms(pred_list[2], processed_gts[2])
            disp_per_pixel_rms_sum += per_pixel_rms_disp
            disp_total_rms_sum += total_rms_disp

            total_loss_sum += pred_list[3]
            for loss_num in range(len(loss_list_sum)):
                loss_list_sum[loss_num] += pred_list[5 + loss_num]
            input_len = input_len + 1
            #if self.epoch_num + 1 == FLAGS.num_of_epchs:
            #    self.Instance_img(set_name, ind, pred_list[1], processed_gts[1])
            if (self.epoch_num+1) % FLAGS.calc_ap_epoch_num == 0:
                Label_ap += self.calc_LabelId_ap(pred_list[0], pred_list[4])

        self.calc_and_set_label_scores(label_id_hist, set_name)
        for loss_num in range(len(loss_list_sum)):
            loss_list_sum[loss_num] = loss_list_sum[loss_num] / input_len
        return_dict = {'labelsID_acc': labelsID_acc_sum / input_len,
                       'InstanceID_per_pixel_rms': InstanceID_per_pixel_rms_sum / input_len,
                       'InstanceID_total_rms': InstanceID_total_rms_sum / input_len,
                       'Disparity_per_pixel_rms': disp_per_pixel_rms_sum / input_len,
                       'Disparity_total_rms': disp_total_rms_sum / input_len,
                       'total_loss': total_loss_sum / input_len,
                       'loss_list': loss_list_sum}
        if (self.epoch_num + 1) % FLAGS.calc_ap_epoch_num == 0:
            return_dict['Label_ap'] = Label_ap / input_len
        return return_dict

    def set_sigmas_and_wights(self):
        if FLAGS.use_multi_loss:
            for i, sigma_sq_tn in zip(range(len(self.multi_loss_class._sigmas_sq)), self.multi_loss_class._sigmas_sq):
                sigma_sq = sigma_sq_tn.eval()
                wight = 1 / (2 * sigma_sq)
                self.statistics['sigmas_list'][i].append(sigma_sq)
                self.statistics['weights_list'][i].append(wight)

    def get_run_list(self, logits):
        run_list = [logits[0], logits[1], logits[2], self.total_loss, self.processed_ground_truths[0]]
        for loss in self.loss_list:
            run_list.append(loss)
        return run_list

    def calc_and_save_examples(self, logits, sess):
        example_inputs, example_ground_truths_many = dh.get_all_data('example')
        for ind, example_input, example_ground_truths in zip(range(len(example_inputs)), example_inputs, example_ground_truths_many):
            example_preds = (sess.run([logits[0], logits[1], logits[2]], feed_dict={self.input_ph: example_input}))
            example_labelsID = self.calc_labelsID_rgb_img(example_preds[0])
            mask = example_ground_truths[1][:, :, :, 2].squeeze(0)
            if FLAGS.need_resize:
                mask = scipy.misc.imresize(mask, (FLAGS.output_height, FLAGS.output_width))
            mask = np.expand_dims(mask, 2)
            example_InstanceID = self.calc_InstanceID_example(example_preds[1].squeeze(0), mask)
            example_Disparity = example_preds[2]
            scipy.misc.imsave(os.path.join(FLAGS.example_preds, 'label', "example_%08d_epoch_%08d.png" % (ind, self.epoch_num)), example_labelsID)
            scipy.misc.imsave(os.path.join(FLAGS.example_preds, 'disp',  "example_%08d_epoch_%08d.png" % (ind, self.epoch_num)), example_Disparity.squeeze())
            if (self.epoch_num + 1) % FLAGS.example_OPTICS_epoch == 0:
                scipy.misc.imsave(os.path.join(FLAGS.example_preds, 'instance', "example_%08d_epoch_%08d.png" % (ind, self.epoch_num)), example_InstanceID[0])
            scipy.misc.imsave(os.path.join(FLAGS.example_preds, 'instance', "y_reg", "example_%08d_epoch_%08d.png" % (ind, self.epoch_num)), example_InstanceID[1])
            scipy.misc.imsave(os.path.join(FLAGS.example_preds, 'instance', "x_reg", "example_%08d_epoch_%08d.png" % (ind, self.epoch_num)), example_InstanceID[2])
            self.save_latest_example(ind, example_labelsID, example_Disparity, example_InstanceID)
        return None

    def save_latest_example(self, ind, example_labelsID, example_Disparity, example_InstanceID):
        scipy.misc.imsave(os.path.join(FLAGS.example_preds, "example_%08d_latest_label.png" % (ind)), example_labelsID)
        scipy.misc.imsave(os.path.join(FLAGS.example_preds, "example_%08d_latest_disp.png" % (ind)), example_Disparity.squeeze())
        if (self.epoch_num + 1) % FLAGS.example_OPTICS_epoch == 0:
            scipy.misc.imsave(os.path.join(FLAGS.example_preds, "example_%08d_latest_instance.png" % (ind)), example_InstanceID[0])
        plt.clf()
        plt.pcolormesh(example_InstanceID[3], cmap='jet')
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(FLAGS.example_preds, 'instance', "y_reg", "example_%08d_latest.png" % (ind)))
        plt.clf()
        plt.pcolormesh(example_InstanceID[4], cmap='jet')
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(FLAGS.example_preds, 'instance', "x_reg", "example_%08d_latest.png" % (ind)))
        np.save(os.path.join(FLAGS.example_preds, 'instance', "y_reg", "example_%08d_latest" % (ind)), example_InstanceID[3])
        np.save(os.path.join(FLAGS.example_preds, 'instance', "x_reg", "example_%08d_latest" % (ind)), example_InstanceID[4])

    def calc_InstanceID_example(self, xy_image, mask):
        raw_image = np.concatenate([xy_image, mask], axis=2)
        cmap = plt.get_cmap('jet')
        y_image = np.delete(cmap(xy_image[:, :, 0]), 3, 2)
        x_image = np.delete(cmap(xy_image[:, :, 1]), 3, 2)

        opt = None
        if (self.epoch_num + 1) % FLAGS.example_OPTICS_epoch == 0:
            opt = OPTICS.calc_clusters_img(raw_image)
        return [opt, y_image, x_image, xy_image[:, :, 0].squeeze(), xy_image[:, :, 1].squeeze()]

    def calc_labelsID_rgb_img(self, label_pred):
        labeled_img = label_pred.squeeze().argmax(axis=2)
        #conc_labeled_img = np.concatenate([np.expand_dims(labeled_img, 2), np.expand_dims(labeled_img, 2)], 2)
        #conc_labeled_img = np.concatenate([conc_labeled_img, np.expand_dims(labeled_img, 2)], 2)
        size = [label_pred.shape[1], label_pred.shape[2]]
        size.append(3)
        rgb_img = np.zeros(size)
        for ind, color in zip(range(len(config.colors[config.working_dataset])), config.colors[config.working_dataset]):
            rgb_img[labeled_img == ind] = color
        return rgb_img

    def calc_labelsID_acc(self, pred, GT):
        gt_labeled_img = GT.squeeze().argmax(axis=2)
        labeled_img = pred.squeeze().argmax(axis=2)
        return np.sum(gt_labeled_img == labeled_img) / (gt_labeled_img.size)

    def calc_InstanceID_rms(self, pred, GT):
        mask = np.expand_dims(GT[:, :, 2], axis=-1)
        num_of_valid_pixels = np.sum(mask)
        mask = np.concatenate([mask, mask], axis=2)
        r_sq_matrix = np.sum(np.square(pred.squeeze()*mask - GT[:, :, 0:2]*mask), axis=-1)
        if num_of_valid_pixels > 0:
            per_pixel_rms = np.sum(np.sqrt(r_sq_matrix))/num_of_valid_pixels
            total_rms = np.sqrt(np.sum(r_sq_matrix)/num_of_valid_pixels)
            return per_pixel_rms, total_rms
        else:
            return 0, 0

    def calc_LabelId_ap(self, pred, GT):
        '''
        Calculating Average Precision (without invalid classes (0-3))
        '''
        return aps(GT[:, :, :, 4:].reshape(-1), pred[:,:,:,4:].reshape(-1))

    def calc_Disparity_rms(self, pred, GT):
        mask = GT[:, :, 1]
        num_of_valid_pixels = np.sum(mask)
        r_sq_matrix = np.sum(np.square(pred.squeeze() * mask - GT[:, :, 0:1].squeeze() * mask), axis=-1)
        if num_of_valid_pixels > 0:
            per_pixel_rms = np.sum(np.sqrt(r_sq_matrix))/num_of_valid_pixels
            total_rms = np.sqrt(np.sum(r_sq_matrix)/num_of_valid_pixels)
            return per_pixel_rms, total_rms
        else:
            return 0, 0

    def get_feed_dict(self, input, outputs):
        feed_dict = {self.input_ph: input}
        feed_dict[self.ground_truths_ph[0]] = outputs[0]
        feed_dict[self.ground_truths_ph[1]] = outputs[1]
        feed_dict[self.ground_truths_ph[2]] = outputs[2]
        return feed_dict

    def save_dict(self, dict_to_save, name):
        f = open(os.path.join(FLAGS.stat_dump_path, name + '.pkl'), "wb")
        pickle.dump(dict_to_save, f)
        f.close()

    def save_plots(self):
        self.save_single_plots([self.statistics['total_loss']['val'], self.statistics['total_loss']['train']],
                               ['val', 'train'], os.path.join(FLAGS.plots_path, 'total_loss'), title='total_loss',
                               ylabel='Total Loss', xlabel='Epoch')

        self.save_single_plots([self.statistics['labelsID_acc']['val'], self.statistics['labelsID_acc']['train']],
                               ['val', 'train'], os.path.join(FLAGS.plots_path, 'labelsID_acc'), title='Labels ID Accuracy',
                               ylabel='Accuracy', xlabel='Epoch')

        self.save_single_plots([self.statistics['InstanceID_per_pixel_rms']['val'], self.statistics['InstanceID_per_pixel_rms']['train']],
                               ['val', 'train'],  os.path.join(FLAGS.plots_path, 'InstanceID_per_pixel_rms'), title='Instance ID per pixel RMS',
                               ylabel='RMS', xlabel='epoch')

        self.save_single_plots([self.statistics['InstanceID_total_rms']['val'], self.statistics['InstanceID_total_rms']['train']],
                               ['val', 'train'], os.path.join(FLAGS.plots_path, 'InstanceID_total_rms'),
                               title='Instance total RMS',
                               ylabel='RMS', xlabel='epoch')

        self.save_single_plots([self.statistics['Disparity_per_pixel_rms']['val'], self.statistics['Disparity_per_pixel_rms']['train']],
                               ['val', 'train'], os.path.join(FLAGS.plots_path, 'Disparity_per_pixel_rms'),
                               title='Disparity ID per pixel RMS',
                               ylabel='RMS', xlabel='epoch')

        self.save_single_plots([self.statistics['Disparity_total_rms']['val'], self.statistics['Disparity_total_rms']['train']],
                               ['val', 'train'], os.path.join(FLAGS.plots_path, 'Disparity_total_rms'),
                               title='Disparity total RMS',
                               ylabel='RMS', xlabel='epoch')

        self.save_dict(self.statistics, 'Statistics_dictionary')

        #----- ap -----
        if (self.epoch_num + 1) % FLAGS.calc_ap_epoch_num == 0:
            # self.save_single_plots([self.Instance_ap_acc_eval['val'],
            #                        self.Instance_ap_acc_eval['train']],
            #                       ['val', 'train'],  FLAGS.plots_path + '/InstanceID_ap',
            #                       title='Instance ID Average Precision', ylabel='Average Percision', xlabel='Epoch')
            self.save_single_plots([self.Label_ap_acc_eval['val'],
                               self.Label_ap_acc_eval['train']],
                              ['val', 'train'], os.path.join(FLAGS.plots_path, 'labelsID_ap'), title='Labels ID Average Precision',
                              ylabel='Average Precision', xlabel='Epoch')
        #--------------



        for loss_num in range(len(self.loss_list)):
            plot_path = os.path.join(FLAGS.plots_path, 'loss_' + str(loss_num)) #+ '_' + str(self.epoch_num)
            self.save_single_plots([self.statistics['loss_lists']['val'][loss_num], self.statistics['loss_lists']['train'][loss_num]],
                                   ['val', 'train'], plot_path,
                                   title='Loss num: ' + str(loss_num), ylabel='loss', xlabel='epoch')

        legend = []
        for i in range(len(self.statistics['sigmas_list'])):
            legend.append('sigma ' + str(i))
        self.save_single_plots(self.statistics['sigmas_list'], legend, os.path.join(FLAGS.plots_path, 'sigmas'), title='Sigmas Sq',
                               ylabel='sigma Sq value', xlabel='epoch')
        legend = []
        list_to_plot = []
        if FLAGS.use_multi_loss:
            for i in range(len(self.statistics['sigmas_list'])):
                legend.append('TU ' + str(i))
                list_to_plot.append(np.array(self.statistics['weights_list'][i])*np.array(self.statistics['loss_lists']['train'][i]+np.log(np.array(self.statistics['sigmas_list'][i]))))
            self.save_single_plots(self.statistics['sigmas_list'], legend, os.path.join(FLAGS.plots_path, 'Task_uncertainty'), title='Task uncertainty',
                                   ylabel='TU', xlabel='epoch')
        legend = []
        for i in range(len(self.statistics['weights_list'])):
            legend.append('weight ' + str(i))
        self.save_single_plots(self.statistics['weights_list'], legend, os.path.join(FLAGS.plots_path, 'weights'), title='Wights',
                               ylabel='wight value', xlabel='epoch')
        scores_name_list = list(self.statistics['label_scores'].keys())
        for name in scores_name_list:
            if name in ['iu', 'cl_acc']:
                continue
            plot_path = os.path.join(FLAGS.plots_path, 'label_acc_' + name)  # + str(self.epoch_num)
            self.save_single_plots([self.statistics['label_scores'][name]['val'], self.statistics['label_scores'][name]['train']],
                                   ['val', 'train'], plot_path, title=name, ylabel=name, xlabel='epoch')
        self.save_dict(self.statistics['label_scores'], 'label_scores')

    def save_single_plots(self, results, legend, plot_path, title='model accuracy', ylabel='accuracy', xlabel='epoch'):
        plt.clf()
        epochs = range(len(results[0]))
        for result in results:
            plt.plot(epochs, result)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(legend, loc='upper right')
        plt.savefig(plot_path + '.png')

    def calc_and_set_label_scores(self, hist_2d, set_name):
        acc, cl_acc_mean, iu_mean, cl_acc, iu = self.get_label_score(hist_2d)
        iu_no_void_mean = self.calc_iu_no_void(iu)
        self.statistics['label_scores']['acc'][set_name].append(acc)
        self.statistics['label_scores']['cl_acc_mean'][set_name].append(cl_acc_mean)
        self.statistics['label_scores']['iu_mean'][set_name].append(iu_mean)
        self.statistics['label_scores']['iu_no_void_mean'][set_name].append(iu_no_void_mean)
        self.statistics['label_scores']['cl_acc'][set_name].append(cl_acc)
        self.statistics['label_scores']['iu'][set_name].append(iu)

    def calc_iu_no_void(self, iu):  #TODO: need to fix
        index_list = []
        for i in range(28):
            if i not in [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 12, 29, 30]:   #TODO: move to config
                index_list.append(i)
        return np.nanmean(iu[index_list])

    def get_label_score(self, hist):
        # Mean pixel accuracy
        acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
        # Per class accuracy
        cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)
        # Per class IoU
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)
        return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu

    def fast_hist(self, a, b, n):
        n = len(config.colors[config.working_dataset])
        a = a.squeeze().argmax(axis=2).flatten()
        b = b.squeeze().argmax(axis=2).flatten()
        k = np.where((a >= 0) & (a < n))[0]
        bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)
        if len(bc) != n ** 2:
            # ignore this example if dimension mismatch
            return 0
        return bc.reshape(n, n)
