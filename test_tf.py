import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict
from collections import Counter

import matplotlib.pyplot as plt

import numpy as np 
import inputs

import tensorflow as tf
import configuration
import model_class

class BasicDataProvider:
  def __init__(self, dataset):
    print('Initializing data provider for dataset %s...' % (dataset, ))

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    print('BasicDataProvider: reading %s' % (dataset_path, ))
    self.dataset = json.load(open(dataset_path, 'r'))

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
    print('BasicDataProvider: reading %s' % (features_path, ))
    features_struct = scipy.io.loadmat(features_path)
    self.features = features_struct['feats']

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)

  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      img['feat'] = self.features[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = random.choice(images)
    sent = random.choice(img['sentences'])

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])

class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.
    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id

def _create_vocab(captions, min_word_count=4):
  """Creates the vocabulary of word to word_id.
  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.
  Args:
    captions: A list of lists of strings.
  Returns:
    A Vocabulary object.
  """
  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  #with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
  #  f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  #print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab

if __name__ == "__main__":
    # Truncate for testing
    n = 10

    dataset = 'flickr8k'
    dp = BasicDataProvider(dataset)

    images = np.array([img for img in dp.iterImages()])[:n]
    captions = np.array([sent['tokens'] for sent in dp.iterSentences()])[:5*n]
    vocab = _create_vocab(captions)
    images_and_captions = []
    for im in images:
        for sent in im['sentences']:
            caption = np.array([vocab.word_to_id(word) for word in sent['tokens']])
            images_and_captions.append( [im['feat'], caption] )

    input_vals = inputs.batch_with_dynamic_pad(images_and_captions,
                                                            batch_size=64,
                                                            queue_capacity=100,
                                                            add_summaries=True)

    image_embeddings, input_seqs, target_seqs, input_mask = input_vals

    model_config = configuration.ModelConfig()
    train_vgg = False
    number_of_steps = 1000

    # Build the TensorFlow graph
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_class.ShowAndTellModel(
            model_config, mode="train", train_vgg=train_vgg)
        # Insert pre-made image embeddings
        model.put_inputs(image_embeddings, input_seqs, target_seqs, input_mask)
        model.build()

        # Set up the learning rate.
        learning_rate_decay_fn = None
        if train_vgg:
          learning_rate = tf.constant(training_config.train_inception_learning_rate)
        else:
          learning_rate = tf.constant(training_config.initial_learning_rate)
          if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
              return tf.train.exponential_decay(
                  learning_rate,
                  global_step,
                  decay_steps=decay_steps,
                  decay_rate=training_config.learning_rate_decay_factor,
                  staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

    # Run training.
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        graph=g,
        global_step=model.global_step,
        number_of_steps=number_of_steps,
        init_fn=model.init_fn,
        saver=saver)

