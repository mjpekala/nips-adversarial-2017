"""  Adversarial attacks (both targeted and non-targeted).

 We originally started with custom codes, fell back to a baseline example 
 to avoid issues with the competition evaluation platform, and then
 eventually re-implemented (quick and dirty) the optimization codes.  
"""

# https://stackoverflow.com/questions/41990014/load-multiple-models-in-tensorflow
# https://stackoverflow.com/questions/42546365/how-to-restore-variables-of-a-particular-scope-from-a-saved-checkpoint-in-tensor
# https://stackoverflow.com/questions/37086268/rename-variable-scope-of-saved-model-in-tensorflow


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import csv
import os
import pdb

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v2

import models.inception_resnet_v2 as ir


import ell_infty_attacks as eia
import classifiers
from classifiers import NETWORKS, NOISY_NETWORKS, VERY_NOISY_NETWORKS


slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
    'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'num_iter', 20, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_string(
    'target_model', '', 'Optional - which network to attack if only one.')

tf.flags.DEFINE_integer(
    'debug', 0, 'Optional debugging outputs.  TURN OFF IF RUNTIME IS LIMITED!')

FLAGS = tf.flags.FLAGS



#----------------------------------------------------------------------

def load_target_class(input_dir):
  """Loads target classes."""
  fn = os.path.join(input_dir, "target_class.csv")

  if not os.path.exists(fn):
    return None

  with tf.gfile.Open(fn) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}



def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath, mode='rb') as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images



def _all_image_names(input_dir):
  "Returns a list of all input files (without having to load images or deal with a generator."
  full_names = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  return [os.path.basename(x) for x in full_names]



def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')




def analyze_attacks(networks_to_use, clean_predictions, target_predictions=None):
  """  Detailed performance metrics for attacks.  This is for debugging only.
  """

  all_files = [x for x in clean_predictions.keys()];  all_files.sort()

  y0 = np.zeros((len(all_files),1), np.int32)
  yt = np.zeros(y0.shape, dtype=y0.dtype)
  y_hat_all = np.zeros((len(all_files), len(networks_to_use)), np.int32)

  #
  # column one is prediction on clean image by some baseline classifier
  #
  for idx, img_name in enumerate(all_files):
    y0[idx] = clean_predictions[img_name]
    yt[idx] = target_predictions[img_name] if target_predictions is not None else -1

  #
  # predictions on AE
  #
  for m_idx, model in enumerate(networks_to_use):
    files, y_hat = model.predict_all_images(FLAGS.output_dir)
    y_hat = np.argmax(y_hat, axis=1)
    tmp_dict = { files[ii] : y_hat[ii] for ii in range(len(files))}

    for idx, img_name in enumerate(all_files):
      y_hat_all[idx, m_idx] = tmp_dict[img_name]

  #
  # performance
  #
  print([p.name for p in networks_to_use])
  print(np.concatenate([y0, yt, y_hat_all], axis=1))

  print('[metrics]:  Number of changed decisions for each classifier')
  print(np.sum(y0 != y_hat_all, axis=0))

  if target_predictions is not None:
    print('[metrics]:  Number of successful targeted attacks')
    print(np.sum(yt == y_hat_all, axis=0))




def _iterative_fast_gradient_attack(ensemble, eps, alpha, num_iter, tic=time.time()):
  """ Constructs the tensorflow graph that implements this attack. 
      Note this does not execute the attack; that is for the caller to do.
  """
  # hardcode since we are in a hurry...
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  # Network inputs
  x_input = tf.placeholder(tf.float32, shape=batch_shape)
  x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
  x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

  target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
  one_hot_target_class = tf.one_hot(target_class_input, num_classes) # note: labels are smoothed in loss function

  # initially x_adv is the same as the input
  x_adv = x_input

  # initialize networks
  if 'InceptionV3' in ensemble:
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      inception.inception_v3(x_input, num_classes=num_classes, is_training=False, scope='InceptionV3')

  if 'resnet_v2_101' in ensemble:
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      resnet_v2.resnet_v2_101(x_input, num_classes=num_classes, is_training=False, scope='resnet_v2_101')
   
  if 'InceptionResnetV2' in ensemble:
    with slim.arg_scope(ir.inception_resnet_v2_arg_scope()):
      ir.inception_resnet_v2(x_input, num_classes=num_classes, is_training=False, scope='InceptionResnetV2')

  if 'Adv-InceptionV3' in ensemble:
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      inception.inception_v3(x_input, num_classes=num_classes, is_training=False, scope='Adv-InceptionV3')

  print('[INFO]: initial setup; net runtime: %0.2f seconds\n' % (time.time()-tic))
  print('[INFO]: ensemble contents:', ensemble)


  for ts in range(num_iter):
    # TODO: this code is gross; clean up if time permits...

    #--------------------------------------------------
    # contribution from InceptionV3
    #--------------------------------------------------
    if 'InceptionV3' in ensemble:
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(x_adv, num_classes=num_classes, is_training=False, reuse=True, scope='InceptionV3')

      cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                      logits,
                                                      label_smoothing=0.1,
                                                      weights=1.0)
      cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                       end_points['AuxLogits'],
                                                       label_smoothing=0.1,
                                                       weights=0.4)
    else:
      cross_entropy = 0

    #--------------------------------------------------
    # contribution from resnet_v2
    #--------------------------------------------------
    if 'resnet_v2_101' in ensemble:
      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_2, _ = resnet_v2.resnet_v2_101(x_adv, num_classes=num_classes, is_training=False, reuse=True, scope='resnet_v2_101')

        # Note: ResnetV2 logits has shape (BATCH_SIZE, 1, 1, N_CLASSES)
        #       Hence the squeeze below.
        cross_entropy_2 = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                           tf.squeeze(logits_2),
                                                           label_smoothing=0.1,
                                                           weights=1.0)
    else:
      cross_entropy_2 = 0

    #--------------------------------------------------
    # contribution from ensemble {resnet_v2, inception_v3}
    #--------------------------------------------------
    if 'InceptionResnetV2' in ensemble:
      with slim.arg_scope(ir.inception_resnet_v2_arg_scope()):
        logits_3, _ = ir.inception_resnet_v2(
          x_adv, num_classes=num_classes, is_training=False,reuse=True, scope='InceptionResnetV2')

      cross_entropy_3 = tf.losses.softmax_cross_entropy(one_hot_target_class, 
                                                       logits_3,
                                                       label_smoothing=0.1,
                                                       weights=1.0)
    else:
      cross_entropy_3 = 0


    #--------------------------------------------------
    # contribution from InceptionV3-adv
    #--------------------------------------------------
    if 'Adv-InceptionV3' in ensemble:
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits_4, end_points_4 = inception.inception_v3(x_adv, num_classes=num_classes, is_training=False, reuse=True, scope='Adv-InceptionV3')

      cross_entropy_4 = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                        logits_4,
                                                        label_smoothing=0.1,
                                                        weights=1.0)
      cross_entropy_4 += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                         end_points_4['AuxLogits'],
                                                         label_smoothing=0.1,
                                                         weights=0.4)
    else:
      cross_entropy_4 = 0


    print('[INFO]: added %d models for timestep %d/%d; net runtime: %0.2f seconds' % (len(ensemble), ts+1, num_iter, time.time()-tic))

    #--------------------------------------------------
    # gradient step
    #--------------------------------------------------
    cross_entropy_avg = (cross_entropy + cross_entropy_2 + cross_entropy_3 + cross_entropy_4) / len(ensemble)

    nabla_x = tf.gradients(cross_entropy_avg, x_adv)[0]

    # EXPERIMENT: add some random noise to gradient (avoid "overfitting"?)
    #if False:
    #  print('[WARNING]: using experimental noisy gradient!')
    #  nabla_x = nabla_x + tf.random_normal(tf.shape(nabla_x), mean=0, stddev=1e-2)

    # EXPERIMENT: avoid moving in directions corresponding to small values of gradient
    #if False:
    #  print('[WARNING]: using experimental gradient clipping!')
    #  # 1e-2 too large of a threshold
    #  nabla_x = tf.where(tf.abs(nabla_x) < 1e-3, tf.zeros(tf.shape(nabla_x)), nabla_x)

    x_next = x_adv - alpha * tf.sign(nabla_x)

    # Always clip at the end
    x_next = tf.clip_by_value(x_next, x_min, x_max)
    x_adv = x_next

  #--------------------------------------------------
  # set up model weight loading
  #--------------------------------------------------
  savers = []
  if 'InceptionV3' in ensemble:
      savers.append((tf.train.Saver(slim.get_model_variables(scope='InceptionV3')), './Weights/inception_v3.ckpt'))
  if 'resnet_v2_101' in ensemble:
      savers.append((tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101')), './Weights/resnet_v2_101.ckpt'))
  if 'InceptionResnetV2' in ensemble:
      savers.append((tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2')), './Weights/ens_adv_inception_resnet_v2.ckpt'))
  if 'Adv-InceptionV3' in ensemble:
      savers.append((tf.train.Saver(slim.get_model_variables(scope='Adv-InceptionV3')), './Weights/InceptionV3-adv.ckpt'))

  print('[INFO]: FG(S)M setup complete; took %0.2f seconds\n' % (time.time()-tic))

  return x_adv, [x_input, target_class_input], savers


#-------------------------------------------------------------------------------
# main functions for various attack types
#-------------------------------------------------------------------------------

def main_fgsm_attack(all_images_target_class, is_targeted):
  """ Runs an iterative attack against input images.
  """
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  alpha = 2.0 * FLAGS.iter_alpha / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  if not is_targeted:
    alpha *= -1

  # which model(s) to attack
  if len(FLAGS.target_model):
    ensemble = [FLAGS.target_model]
  else:
    # default is to attack everything
    #ensemble = ['InceptionV3', 'resnet_v2_101', 'InceptionResnetV2', 'Adv-InceptionV3']
    ensemble = ['InceptionV3', 'InceptionResnetV2', 'Adv-InceptionV3']

  tic = time.time()

  with tf.Graph().as_default():
    # setup attack graph
    x_adv, (x_in, y_target_in), savers = _iterative_fast_gradient_attack(ensemble, eps, alpha, FLAGS.num_iter, tic)

    # do the computation
    session_creator = tf.train.ChiefSessionCreator(master=FLAGS.master)
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      # load model parameters
      for saver, weight_file in savers:
        saver.restore(sess, weight_file)

      # do attack
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        target_class_for_batch = (
            [all_images_target_class[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames)))
        adv_images = sess.run(x_adv,
                              feed_dict={x_in : images, 
                                         y_target_in : target_class_for_batch})
        save_images(adv_images, filenames, FLAGS.output_dir)



def main_optimization_attack(all_images_target_class, is_targeted):
  """ Executes optimization-based attack against input images.
  """
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  tic = time.time()

  with tf.Graph().as_default():
    with tf.Session() as sess:

      # for now we attack one big graph; in the future, could attack each network
      # separately and then aggregate.
      #
      # NOTE: if we had a larger time budget, attacking the noisy networks can provide some
      #       robustness to additive Gaussian noise defense.  However, if time is very
      #       limited (as we anticipate) then attacking noisy networks slows down our 
      #       convergence vs. clean networks which is probably not worth the trade given
      #       we don't know how many will be playing Gaussian noise defense.
      #
      # 
      #f_logit = lambda x: .20 * NOISY_NETWORKS[0]._prediction_graph_output(x, pre_softmax=True) + \
      #                    .20 * NOISY_NETWORKS[1]._prediction_graph_output(x, pre_softmax=True) + \
      #                    .20 * NOISY_NETWORKS[2]._prediction_graph_output(x, pre_softmax=True) + \
      #                    .40 * NOISY_NETWORKS[3]._prediction_graph_output(x, pre_softmax=True)
      f_logit = lambda x: .20 * NETWORKS[0]._prediction_graph_output(x, pre_softmax=True) + \
                          .20 * NETWORKS[2]._prediction_graph_output(x, pre_softmax=True) + \
                          .60 * NETWORKS[3]._prediction_graph_output(x, pre_softmax=True)

      # construct attack graph
      # NOTE: will correct t_max later/below.
      #
      opts = eia.AttackOptions(learn_rate=eps/10., tau=eps, c=0.1, t_max=.5, sigma=0.0, n_restarts=1)
      attacker = eia.AdamAttack(opts, f_logit, batch_shape, is_targeted)

      # load model weights (now that graph exists)
      # need to keep this in sync with f_logit...
      #for n_id in [0,1,2,3]:
      for n_id in [0,2,3]:
        saver = tf.train.Saver(slim.get_model_variables(scope=NETWORKS[n_id]._scope))
        saver.restore(sess, NETWORKS[n_id]._weights_file)

      # run one pass just to get things warmed up (first iteration usually seems slower...)
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        _ = attacker.attack(sess, images, np.array([100,] * images.shape[0], np.float32))
        break

      # estimate how much time we have left
      # total time is 500s; we shave some of that off to be safe
      NET_TIME = 500.0
      elapsed = (time.time() - GLOBAL_START_TIME)
      est_time_remaining = (NET_TIME - 19) - elapsed 

      # *** WARNING WARNING WARNING ***
      #   Based on cleverhans codes, it seems the list of target labels may be larger than 100!! 
      #   So, do NOT base time calculation on size of that list!!!
      # *** WARNING WARNING WARNING ***
      #n_batches = int(np.ceil(1.0 * len(all_images_target_class) / batch_shape[0]))
      #secs_per_batch = est_time_remaining / n_batches 
      n_batches = int(np.ceil(100.0 / batch_shape[0]))
      secs_per_batch = est_time_remaining / n_batches

      opts2 = eia.AttackOptions(t_max=secs_per_batch, 
          learn_rate=opts.learn_rate, tau=opts.tau, c=opts.c, sigma=opts.sigma, n_restarts=opts.n_restarts)
      attacker.opts = opts2
      print('[info]: elapsed: %0.2f sec.; remaining: %0.2f; # batches: %d; using %0.2f sec/batch' % (elapsed, est_time_remaining, n_batches, secs_per_batch))

      # run the attack
      total_cnt = 0
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        target_class_for_batch = (
            [all_images_target_class[n] for n in filenames]
            + [0] * (batch_shape[0] - len(filenames)))

        x_adv_raw, loss1, loss2, pred_final = attacker.attack(sess, images, target_class_for_batch)

        # enforce hard constraint on max perturbation
        # (optimizer has this as a soft constraint)
        delta = images - x_adv_raw
        delta[delta < -eps] = -eps
        delta[delta > eps] = eps
        x_adv = images - delta

        save_images(x_adv, filenames, FLAGS.output_dir)

        total_cnt += len(filenames)
        print('[info]: processed %d images; net runtime %0.2f' % (total_cnt, time.time() - GLOBAL_START_TIME))



def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #--------------------------------------------------
  # Determine target labels 
  #--------------------------------------------------
  all_images_target_class = load_target_class(FLAGS.input_dir)

  if all_images_target_class is None:
    print('[INFO]: this is a NON-TARGETED attack with tau=%d' % FLAGS.max_epsilon)

    if False:
      # Heuristic 1: push estimate away from that of original/clean prediction
      files_0, y_hat_0 = NETWORKS[0].predict_all_images(FLAGS.input_dir)
      y_hat_0 = np.argmax(y_hat_0, axis=1)
      all_images_target_class = {files_0[ii] : y_hat_0[ii] for ii in range(len(files_0))}
      is_targeted = False
    else:
      # Heuristic 2: randomly select a class and do targeted attack
      # For an ensemble of networks, this empirically seems to work better in some cases.
      print('[INFO]: converting NON-TARGETED into a TARGETED attack...')
      files_0 = _all_image_names(FLAGS.input_dir)
      all_images_target_class = {files_0[ii] : np.random.choice(1000) + 1 for ii in range(len(files_0))}
      is_targeted = True

  else:
    print('[INFO]: this is a TARGETED attack with tau=%d, # targets: %d' % (FLAGS.max_epsilon, len(all_images_target_class)))
    is_targeted = True 


  #--------------------------------------------------
  # launch desired attack
  #--------------------------------------------------
  main_optimization_attack(all_images_target_class, is_targeted)
  #main_fgsm_attack(all_images_target_class, is_targeted)


  #--------------------------------------------------
  # post-processing analysis (optional) 
  #--------------------------------------------------
  if FLAGS.debug:
    # predictions on clean data
    files_0, y_hat_0 = NETWORKS[0].predict_all_images(FLAGS.input_dir)
    y_hat_0 = np.argmax(y_hat_0, axis=1)
    clean_predictions = { files_0[ii] : y_hat_0[ii] for ii in range(len(files_0))}

    # compare to AE predictions
    if is_targeted:
      analyze_attacks(NETWORKS, clean_predictions, all_images_target_class)
      analyze_attacks(VERY_NOISY_NETWORKS, clean_predictions, all_images_target_class)
    else:
      analyze_attacks(NETWORKS, clean_predictions)
      analyze_attacks(VERY_NOISY_NETWORKS, clean_predictions)

  print('[info]: Done! Total runtime: %0.2f (secs)' % (time.time() - GLOBAL_START_TIME))


if __name__ == '__main__':
  GLOBAL_START_TIME = time.time()
  tf.app.run()
