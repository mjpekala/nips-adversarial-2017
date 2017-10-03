""" Ensemble defense with gradient-based confidence measure.

  We were having so much difficulty getting the competition to evaluate our 
  earlier (keras-based) submissions that we just started over and used exactly 
  a sample defense as a starting point.  This set us back in a number of 
  respects but hopefully this code will run more reliably on the remote cloud.

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__author__ = "mjp"
__date__ = "sept. 2017"


import time
import csv
import os
import glob
import inspect
import pdb

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.stats import ks_2samp

import tensorflow as tf
#from tensorflow.contrib.slim.nets import inception
#from tensorflow.contrib.slim.nets import resnet_v2
#import ens_adv_inception_resnet_v2.inception_resnet_v2 as ir

import classifiers


slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output filename.')



FLAGS = tf.flags.FLAGS

# hard coded for now because we are in a hurry...
NUM_CLASSES = 1001
BATCH_SHAPE = [16, 299, 299, 3]
BASELINE_DIR = './Baselines'
RNG_SEED = 1066
LABEL_SMOOTHING=0.1



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

  all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))

  #for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
  for filepath in all_files:
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



def smooth_one_hot_predictions(p):
  """Given a vector of predicted class labels p, generates a 'smoothed' one-hot prediction matrix."""
  out = (1./NUM_CLASSES) * np.ones((p.size, NUM_CLASSES), dtype=np.float32)
  for ii in range(p.size):
    out[ii,p[ii]] = 0.9
  return out


def gradient_norm(G):
  """ Computes two norm of the tensor G whose first dimension is assumed to be # of examples. """
  out = G ** 2
  while (out.ndim > 1):
    out = np.sum(out, axis=out.ndim-1)
  return np.sqrt(out)


def compare(samp1, samp2):
  d, p_value = ks_2samp(samp1, samp2)
  return p_value


def check_baseline_agreement():
  files = glob.glob(os.path.join(BASELINE_DIR, '*npz'))
  preds = []
  for fn in files:
    preds.append(np.load(fn)['preds'])

  P = (np.c_[preds]).transpose()

  in_agreement = np.all(P[:,[0]] == P, axis=1)
  print('[INFO]: %d classifiers agree on %0.2f%% of class estimates' % (P.shape[1], 100.*np.sum(in_agreement) / in_agreement.size))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tic = time.time()

  print('[INFO]: will evaluate images in: "%s"' % FLAGS.input_dir)
  print('[INFO]: writing output to:       "%s"' % FLAGS.output_file)

  models = [
           classifiers.ResnetV2('resnet_v2_101', './Weights/resnet_v2_101.ckpt'),
           classifiers.InceptionV3('InceptionV3', './Weights/inception_v3.ckpt'),
           classifiers.InceptionV3('Adv-InceptionV3', './Weights/InceptionV3-adv.ckpt'),
           classifiers.AdvResnetV2InceptionV3('InceptionResnetV2', './ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt'),
           ]

  if len(FLAGS.output_file) == 0:
    #--------------------------------------------------
    # Generating new baseline statistics
    #--------------------------------------------------
    print('[INFO]: generating baseline statistics')

    for model in models:
      filenames, preds, conf = model(FLAGS.input_dir, save_baseline=True)

    check_baseline_agreement()

  else:
    #--------------------------------------------------
    # Evaluating test data (assumes baseline data already generated)
    #--------------------------------------------------
    print('[INFO]: generating class estimates (ie. deploy)')

    best_score = -np.inf
    best_preds = None
    best_filenames = None
    all_scores = []

    # figure out which classifier's estimates to use
    for f in models:
      print('[INFO]: evaluating model', f.name)
      filenames, preds, conf = f(FLAGS.input_dir)

      # (optional) save results for subsequent debugging
      if False:
        debug_fn = 'debugging_' + f.name + ".npz"
        np.savez(debug_fn, conf=conf, preds=preds)

      # load baseline statistics
      base_fn = os.path.join(BASELINE_DIR, f.name + '.npz')
      conf_base = np.load(base_fn)['conf']
      score = compare(conf_base, conf)

      if score > best_score:
        print('[INFO]: score %0.2g improves upon best so far' % (score))
        best_score, best_preds, best_filenames = score, preds, filenames
      else:
        print('[INFO]: score %0.2g does not provide improvement' % (score))
      all_scores.append(score)

    # save "best" predictions to file
    with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
      for ii in range(len(best_filenames)): 
        out_file.write('{0},{1}\n'.format(best_filenames[ii], best_preds[ii]))

    # show all scores (for debugging)
    for ii, f in enumerate(models):
        selected_str = '  (*)' if all_scores[ii] == best_score else ''
        print('[INFO]: ', all_scores[ii], ' for network ', f.name, selected_str)


if __name__ == '__main__':
  tf.app.run()
