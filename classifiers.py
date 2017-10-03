""" Wrappers around standard deep networks used by attacks & defenses.
"""

__author__ = "mjp"
__email__ =  "mpekala@umd.edu"
__date__ = "september, 2017"
__license__ = "Apache 2.0"


import time, os
import pdb

import numpy as np

from scipy.misc import imread, imsave

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2, vgg

import models.inception_v4 as iv4
import models.inception_resnet_v2 as ir

slim = tf.contrib.slim





#-------------------------------------------------------------------------------

def _input_filenames(input_dir):
  all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  all_files.sort()
  return all_files


def _load_images(input_dir, batch_shape):
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

  for filepath in _input_filenames(input_dir):
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


#-------------------------------------------------------------------------------

class Generic_Network:
    def __init__(self, scope, weights_file, 
                 shape=[10, 299, 299, 3], 
                 num_classes=1001,
                 sigma=0.0,  # for networks that know how to add noise
                 tf_master=''):
      """ Initialize network properties. """

      self._scope = scope
      self._weights_file = weights_file

      self._shape = shape
      self._tf_master = tf_master
      self._num_classes = num_classes

      self._label_smoothing = 0.1
      self._baseline_dir = './Baselines'
      self._sigma=sigma

      self.name = self._scope


    def _baseline_filename(self):
      return os.path.join(self._baseline_dir, 'STATS_'+self.name+'.npz')


    def __call__(self, *args, **kargs):
      "This is just some convenient shorthand."
      return self.predict_all_images_with_confidence(*args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=False):
      """ Returns a tensorflow variable representing the (scalar) predictions 
          (ie. prediction outputs).
      """
      raise RuntimeError('this method should be overridden!')


    def _dloss_dx(self, x_input, y_hat_input):
      """ Returns a tensorflow variable representing the derivative of the loss w.r.t. the input.
          This is evaluated at (x,y)
      """
      raise RuntimeError('this method should be overridden!')


    def predict_all_images(self, input_dir, num_samples=1):
      """ Generates predictions (only) for all images in the input directory.

      If you also want confidence scores, use predict_all_images_with_confidence() instead.
      """
      tic = time.time()

      n_images = len(_input_filenames(input_dir))
      predictions_all = np.zeros((n_images, self._num_classes))
      files_all = []

      with tf.Graph().as_default():
        #
        # All models must take the same input.
        # The output variable depends upon the specific network.
        #
        x_input = tf.placeholder(tf.float32, shape=self._shape)
        y_hat_output = self._prediction_graph_output(x_input, pre_softmax=False)

        saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
        session_creator = tf.train.ChiefSessionCreator(master=self._tf_master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
          saver.restore(sess, self._weights_file)

          idx = 0
          for filenames, images in _load_images(input_dir, self._shape):
            for n in range(num_samples):
              preds = sess.run(y_hat_output, feed_dict={x_input : images})
              preds = preds[0:len(filenames)]  # note: the last mini-batch may have fewer images

              # compute running average
              if n == 0:
                preds_avg = preds
              else:
                preds_avg = preds_avg + (preds - preds_avg) / n

            files_all.extend(filenames)
            predictions_all[idx:(idx+len(filenames)),:] = preds_avg
            idx += len(filenames)

      print('[%s]: time to predict %d images: %0.2f (sec)' % (self.name, len(files_all), time.time()-tic))
      return files_all, predictions_all



    def predict_all_images_with_confidence(self, input_dir, save_baseline=False):
      """ Generates class predictions and associated confidence scores.
      """

      #--------------------------------------------------
      # STEP 1 : Initial predictions
      #--------------------------------------------------
      # First, generate predictions on all images.
      files_all, y_hat_all = self.predict_all_images(input_dir)

      # XXX: I don't think we want to use smoothed predictions here...
      #      For this application, we want the "native" prediction?
      #if y_hat_all.ndim == 1:
      #  print('[INFO]:  ** WARNING ** using smoothed predictions!!')
      #  y_hat_all = smooth_one_hot_predictions(y_hat_all)

      #--------------------------------------------------
      # STEP 2 : "Confidence" estimates
      #--------------------------------------------------
      y_hat_minibatch = np.zeros((self._shape[0], self._num_classes), np.float32)
      conf_all = []
      tic = time.time()

      with tf.Graph().as_default():
        # All models must take the same input
        x_input = tf.placeholder(tf.float32, shape=self._shape)
        y_hat_input = tf.placeholder(tf.float32, shape=[self._shape[0], self._num_classes])

        # The gradient calculation depends upon the underlying graph
        nabla_x = self._dloss_dx(x_input, y_hat_input)

        # Evaluate
        saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
        session_creator = tf.train.ChiefSessionCreator(master=self._tf_master)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
          saver.restore(sess, self._weights_file)

          idx = 0
          for filenames, images in _load_images(input_dir, self._shape):
            # grab corresponding subset of predictions
            y_hat_minibatch[0:len(filenames),:] = y_hat_all[idx:(idx+len(filenames)),:]
            idx += len(filenames)

            feed_dict = {x_input : images, y_hat_input : y_hat_minibatch}

            grad = sess.run(nabla_x, feed_dict=feed_dict)
            grad = grad[0:len(filenames),...]  # Note: last batch may contain fewer images
            conf_all.append(gradient_norm(grad))

      print('[%s]: time to generate confidence scores for %d images: %0.2f (sec)' % (self.name, len(files_all), time.time()-tic))

      # reshape into a single tensor 
      conf_all = np.concatenate(conf_all)

      if save_baseline:
        np.savez(self._baseline_filename(), conf=conf_all, preds=np.argmax(y_hat_all, axis=1))
        print('[%s]: saved %d predictions to baseline file' % (self.name, conf_all.size))

      return files_all, y_hat_all, conf_all



    def confidence_quantile(self, v):
      """ Compares confidence scores v to those in the baseline.
      """
      if os.path.exists(self._baseline_file()):
        conf = np.load(self._baseline_file())['conf']
        quantiles = np.zeros((v.size,))
        for ii in range(v.size):
          quantiles[ii] = 1.0 * np.sum(v[ii] > conf) / conf.size
        return quantiles

      else:
        return np.nan


#-------------------------------------------------------------------------------


class InceptionV3(Generic_Network):
    def __init__(self, *args, **kargs):
        Generic_Network.__init__(self, *args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=False):
      if self._sigma > 0:
        x_input = x_input + tf.random_normal(tf.shape(x_input), mean=0, stddev=self._sigma)

      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      if pre_softmax:
        return logits
      else:
        # note: predictions seem to be "probabilities" (softmax outputs)
        #return tf.nn.softmax(logits)
        return end_points['Predictions']


    def _dloss_dx(self, x_input, y_hat_input):
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      #cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input,
      #                                                logits,
      #                                                label_smoothing=0.1,
      #                                                weights=1.0)
      #cross_entropy += tf.losses.softmax_cross_entropy(y_hat_input,
      #                                                 end_points['AuxLogits'],
      #                                                 label_smoothing=0.1,
      #                                                 weights=0.4)

      cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input,
                                                      #logits,
                                                      end_points['Predictions'],
                                                      label_smoothing=self._label_smoothing,
                                                      weights=1.0)

      nabla_x = tf.gradients(cross_entropy, x_input)[0]
      return nabla_x



class InceptionV4(Generic_Network):
    def __init__(self, scope='InceptionV4', weight_file='./Weights/inception_v4.ckpt', *args, **kargs):
        Generic_Network.__init__(self, scope, weight_file, *args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=True):
      if self._sigma > 0:
        x_input = x_input + tf.random_normal(tf.shape(x_input), mean=0, stddev=self._sigma)

      with slim.arg_scope(iv4.inception_v4_arg_scope()):
        logits, end_points = iv4.inception_v4(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      if pre_softmax:
        return logits
      else:
        # note: predictions seem to be "probabilities" (softmax outputs)
        return tf.nn.softmax(logits)
        #return end_points['Predictions']


    def _dloss_dx(self, x_input, y_hat_input):
      with slim.arg_scope(iv4.inception_v4_arg_scope()):
        logits, end_points = iv4.inception_v4(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input,
                                                      logits,
                                                      label_smoothing=self._label_smoothing,
                                                      weights=1.0)

      nabla_x = tf.gradients(cross_entropy, x_input)[0]
      return nabla_x

#-------------------------------------------------------------------------------


class ResnetV2(Generic_Network):
    def __init__(self, *args, **kargs):
        Generic_Network.__init__(self, *args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=False):
      if self._sigma > 0:
        x_input = x_input + tf.random_normal(tf.shape(x_input), mean=0, stddev=self._sigma)

      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      # note: logits have shape [BATCH_SIZE, 1, 1, NUM_CLASSES]
      # Hence the squeeze below.  Assumes BATCH_SIZE > 1.
      logits = tf.squeeze(logits)

      if pre_softmax:
        return logits
      else:
        return tf.nn.softmax(logits)


    def _dloss_dx(self, x_input, y_hat_input):
      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input,
                                                      tf.squeeze(logits),  # note: squeeze needed here
                                                      label_smoothing=self._label_smoothing,
                                                      weights=1.0)

      nabla_x = tf.gradients(cross_entropy, x_input)[0]

      return nabla_x


#-------------------------------------------------------------------------------


class VGG16(Generic_Network):
    def __init__(self, scope='vgg_16', weights_file='./Weights/vgg_16.ckpt', *args, **kargs):
        Generic_Network.__init__(self, scope, weights_file, *args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=True):
      if self._sigma > 0:
        x_input = x_input + tf.random_normal(tf.shape(x_input), mean=0, stddev=self._sigma)

      #
      # make inception-like input work for VGG
      #
      x_input_vgg = tf.image.resize_images(x_input, [224, 224])
      x_input_vgg = (x_input_vgg + 1) * (255./2)                  # [-1,1] -> [0,255]
      x_input_vgg = x_input_vgg[..., ::-1]                        # RGB ->BGR
      x_input_vgg = x_input_vgg - [103.939, 116.8, 123.7]         # subtract mean

      with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(x_input_vgg, 
                               num_classes=1000, # for compatabillity with weights
                               is_training=False, 
                               scope=self._scope, 
                               spatial_squeeze=True)

      # map classes:  1000 -> 1001
      dummy_class = tf.zeros((self._shape[0], 1), tf.float32)
      logits_1001 = tf.concat([dummy_class, logits], axis=1)

      if pre_softmax:
        return logits_1001
      else:
        return tf.nn.softmax(logits_1001)


    def _dloss_dx(self, x_input, y_hat_input):
      raise RuntimeError('todo')

#-------------------------------------------------------------------------------


class AdvResnetV2InceptionV3(Generic_Network):
  """ Adversarially trained ensemble of {ResnetV2, InceptionV3} provided by competition organizers.
  """
  def __init__(self, 
               scope='InceptionResnetV2', 
               weights_file='./Weights/ens_adv_inception_resnet_v2.ckpt', 
               *args, **kargs):
    Generic_Network.__init__(self, scope, weights_file, *args, **kargs)


  def _prediction_graph_output(self, x_input, pre_softmax=False):
    if self._sigma > 0:
      x_input = x_input + tf.random_normal(tf.shape(x_input), mean=0, stddev=self._sigma)

    with slim.arg_scope(ir.inception_resnet_v2_arg_scope()): 
      logits, _ = ir.inception_resnet_v2(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

    if pre_softmax:
      return logits
    else:
      return tf.nn.softmax(logits)


  def _dloss_dx(self, x_input, y_hat_input):
    with slim.arg_scope(ir.inception_resnet_v2_arg_scope()): 
      logits, _ = ir.inception_resnet_v2(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

    cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input, 
                                                    logits,
                                                    label_smoothing=self._label_smoothing,
                                                    weights=1.0)
    nabla_x = tf.gradients(cross_entropy, x_input)[0]

    return nabla_x


#----------------------------------------------------------------------
#
# A canonical list of networks
#
#----------------------------------------------------------------------
NETWORKS = [
            InceptionV3('InceptionV3', './Weights/inception_v3.ckpt'), 
            ResnetV2('resnet_v2_101', './Weights/resnet_v2_101.ckpt'),
            InceptionV3('Adv-InceptionV3', './Weights/InceptionV3-adv.ckpt'),  # adversarially trained
            AdvResnetV2InceptionV3('InceptionResnetV2', './Weights/ens_adv_inception_resnet_v2.ckpt'),
            VGG16(),
            InceptionV4(),
           ]


NOISY_NETWORKS = [
            InceptionV3('InceptionV3', './Weights/inception_v3.ckpt', sigma=.05), 
            ResnetV2('resnet_v2_101', './Weights/resnet_v2_101.ckpt', sigma=.05),
            InceptionV3('Adv-InceptionV3', './Weights/InceptionV3-adv.ckpt', sigma=.05),
            AdvResnetV2InceptionV3('InceptionResnetV2', './Weights/ens_adv_inception_resnet_v2.ckpt', sigma=.05),
            VGG16(sigma=0.05),
            InceptionV4(sigma=.05),
           ]

VERY_NOISY_NETWORKS = [
            InceptionV3('InceptionV3', './Weights/inception_v3.ckpt', sigma=.13), 
            ResnetV2('resnet_v2_101', './Weights/resnet_v2_101.ckpt', sigma=.13),
            InceptionV3('Adv-InceptionV3', './Weights/InceptionV3-adv.ckpt', sigma=.13),
            AdvResnetV2InceptionV3('InceptionResnetV2', './Weights/ens_adv_inception_resnet_v2.ckpt', sigma=.13),
            VGG16(sigma=.13),
            InceptionV4(sigma=.13),
           ]
