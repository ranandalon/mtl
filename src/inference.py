from tqdm import tqdm
import config
import user_config
from utils import utils, data_handler as dh
import tensorflow as tf
import os
import scipy.misc
from PIL import Image
import numpy as np

FLAGS = tf.app.flags.FLAGS
INF_FLAGS = {'use_label_type': True, 'use_label_inst': False, 'use_label_disp': False}
results_dir = os.path.join(os.path.dirname(__file__), 'alon_resNet_label_v3')
inference_results_dir = os.path.join(os.path.dirname(__file__), 'resNet_inference_label')


def inference():
    make_dirs()
    # create place holder img
    input_ph, ground_truths_ph, ground_truths, pre_processed_input = dh.get_place_holders()
    autoencoder = utils.get_autoencoder(user_config.autoencoder, config.working_dataset, config.strided)
    logits = autoencoder.inference(pre_processed_input)
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    num_of_inference_images = FLAGS.num_of_val_imgs  #TODO: more generic num
    input_img, _ = dh.init_data(num_of_inference_images)
    with tf.Session(config=session_config) as sess:
        start_graph(sess, autoencoder, saver)
        summary_writer = tf.summary.FileWriter(os.path.join(inference_results_dir, 'logs'), sess.graph)
        for ind in tqdm(range(num_of_inference_images)):
            input_img[ind], gt = dh.get_data(ind, 'val')  #TODO: more generic name
            run_list = get_run_list(logits)
            pred_list = sess.run(run_list, feed_dict={input_ph: input_img[ind]})
            pred_dict = pred_list2dict(pred_list)
            save_images(pred_dict, ind=ind)


def get_run_list(logits):
    run_list = []
    if INF_FLAGS['use_label_type']:
        labelId_image_gray = utils.labelId(logits[0])
        run_list.append(tf.cast(labelId_image_gray, tf.uint8))
    if INF_FLAGS['use_label_inst']:
        run_list.append(logits[1])
    if INF_FLAGS['use_label_disp']:
        run_list.append(logits[2])
    return run_list


def pred_list2dict(pred_list):
    pred_dict = {}
    if INF_FLAGS['use_label_disp']:
        pred_dict['disp'] = pred_list.pop().squeeze()
    if INF_FLAGS['use_label_inst']:
        pred_dict['instance'] = pred_list.pop().squeeze()
    if INF_FLAGS['use_label_type']:
        pred_dict['label'] = pred_list.pop().squeeze()
    return pred_dict


def save_images(pred_dict, ind=0):
    if INF_FLAGS['use_label_type']:
        save_label(pred_dict['label'], ind=ind)
    if INF_FLAGS['use_label_inst']:
        save_instance(pred_dict['instance'])
    if INF_FLAGS['use_label_disp']:
        save_disp(pred_dict['disp'])


def save_label(gray_scale_img, ind=0):
    scipy.misc.imsave(os.path.join(inference_results_dir, 'label', '%08d.png' % ind), gray_scale_img)


def save_instance(instance_yx, ind=0):
    np.save(os.path.join(inference_results_dir, 'instance', '%08d.png' % ind), instance_yx)


def save_disp(disp_img, ind=0):
    im = Image.fromarray(disp_img)
    im.save(os.path.join(inference_results_dir, 'disp', '%08d.png' % ind), disp_img)


def make_dirs():
    if not os.path.exists(inference_results_dir):
        os.makedirs(inference_results_dir)
    if not os.path.exists(os.path.join(inference_results_dir, 'logs')):
        os.makedirs(os.path.join(inference_results_dir, 'logs'))
    if not os.path.exists(os.path.join(inference_results_dir, 'label')):
        os.makedirs(os.path.join(inference_results_dir, 'label'))
    if not os.path.exists(os.path.join(inference_results_dir, 'instance')):
        os.makedirs(os.path.join(inference_results_dir, 'instance'))
    if not os.path.exists(os.path.join(inference_results_dir, 'disp')):
        os.makedirs(os.path.join(inference_results_dir, 'disp'))


def start_graph(sess, autoencoder, saver):
    checkpoint = tf.train.get_checkpoint_state(os.path.join(results_dir, 'global_ckpt'))
    if not checkpoint:
        raise NameError("Oops! No valid checkpoint path")
    if user_config.autoencoder == 'ResNet':
        autoencoder.restore(sess, restore_first=False, saver=saver, checkpoint=checkpoint)
    else:
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(sess, checkpoint_path)

inference()


