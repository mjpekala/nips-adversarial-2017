"""  Splits the 1000 training images into a 'train' and 'test' pools.
"""

import sys, os
from shutil import copyfile
import numpy as np
import tensorflow as tf
import pdb


if __name__ == "__main__":
  input_dir = sys.argv[1]
  n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 600
  rng_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1097

  train_dir, test_dir, test_dir_2 = './NIPS_1000/Train', './NIPS_1000/Test', './NIPS_1000/Test_Targeted'
  target_class = 100

  np.random.seed(rng_seed)

  # create output directories, if needed
  for dirname in [train_dir, test_dir, test_dir_2]:
    if not os.path.exists(dirname):
      os.makedirs(dirname)

  # copy files
  all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  train_indices = np.random.choice(len(all_files), n_train, replace=False)

  target_dict = {}
  for ii in range(len(all_files)):
    fn = os.path.split(all_files[ii])[-1]

    if ii in train_indices:
      copyfile(all_files[ii], os.path.join(train_dir, fn))
    else:
      copyfile(all_files[ii], os.path.join(test_dir, fn))
      copyfile(all_files[ii], os.path.join(test_dir_2, fn))

      # choose a target class (randomly)
      target_dict[fn] = 1 + np.random.choice(1000)


  with open(os.path.join(test_dir_2, 'target_class.csv'), 'w') as f:
    for key, value in target_dict.iteritems():
      f.write('%s, %d\n' % (key, value))

