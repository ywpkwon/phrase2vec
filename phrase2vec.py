from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from word2vec import Word2Vec
from tensorflow.models.embedding import gen_word2vec as word2vec

from ConceptManager import ConceptManager as CM


flags = tf.app.flags

# flags.DEFINE_string("save_path", None, "Directory to write the model and "
#                     "training summaries.")
# flags.DEFINE_string("train_data", None, "Training text file. "
#                     "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
# flags.DEFINE_string(
#     "eval_data", None, "File consisting of analogies of four tokens."
#     "embedding 2 - embedding 1 + embedding 3 should be close "
#     "to embedding 4."
#     "See README.md for how to get 'questions-words.txt'.")
# flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
# flags.DEFINE_integer(
#     "epochs_to_train", 15,
#     "Number of epochs to train. Each epoch processes the training data once "
#     "completely.")
# flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
# flags.DEFINE_integer("num_neg_samples", 100,
#                      "Negative samples per training example.")
# flags.DEFINE_integer("batch_size", 16,
#                      "Number of training examples processed per step "
#                      "(size of a minibatch).")
# flags.DEFINE_integer("concurrent_steps", 12,
#                      "The number of concurrent training steps.")
# flags.DEFINE_integer("window_size", 5,
#                      "The number of words to predict to the left and right "
#                      "of the target word.")
# flags.DEFINE_integer("min_count", 5,
#                      "The minimum number of word occurrences for it to be "
#                      "included in the vocabulary.")
# flags.DEFINE_float("subsample", 1e-3,
#                    "Subsample threshold for word occurrence. Words that appear "
#                    "with higher frequency will be randomly down-sampled. Set "
#                    "to 0 to disable.")
# flags.DEFINE_boolean(
#     "interactive", False,
#     "If true, enters an IPython interactive session to play with the trained "
#     "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
#     "model.nearby([b'proton', b'elephant', b'maxwell'])")
# flags.DEFINE_integer("statistics_interval", 5,
#                      "Print statistics every n seconds.")
# flags.DEFINE_integer("summary_interval", 5,
#                      "Save training summary to file every n seconds (rounded "
#                      "up to statistics interval).")
# flags.DEFINE_integer("checkpoint_interval", 600,
#                      "Checkpoint the model (i.e. save the parameters) every n "
#                      "seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS





class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.
    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # How often to print statistics.
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file (rounds up to the nearest
    # statistics_interval).
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints (rounds up to the nearest statistics
    # interval).
    self.checkpoint_interval = FLAGS.checkpoint_interval

    # Where to write out summaries.
    self.save_path = FLAGS.save_path
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    # Eval options.
    # The text file for eval.
    self.eval_data = FLAGS.eval_data


# batch_size = 3
# def generate_batch(batch_size, num_skips, skip_window):
#     global data_index

#     # just fixed examples for now
#     phrases = np.array([0, 0, 0], dtype=np.int32)
#     words = np.array([123, 432, 234], dtype=np.int32)
#     labels = np.array([546, 456, 233], dtype=np.int32).reshape([batch_size, 1])
#     return phrases, words, labels

def generate_batch(batch_size, window_size):
    """Generate batch

    Returns:
    para_examples, word_examples, labels
    para_examples:[para_id]
    word_examples:[word_id*(window_size-1)]
    labels: word_id
    """ 

    #para_examples: [para_id]
    para_examples = np.ndarray(shape=(batch_size,1), dtype=np.int32)

    #word_examples: [word_id*(window_size-1)]
    word_examples = np.ndarray(shape=(batch_size,window_size - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    paragraph = self.concept_list[self.para_index].fullConcept()
    for i in range(batch_size):
        
        # if there is enough words for this sample
        while (self.word_index + window_size) > len(paragraph):
            self.para_index = (self.para_index + 1) % len(self.concept_list)
            self.word_index = 0
            paragraph = self.concept_list[self.para_index].fullConcept()
         
        para_examples[i][0] = self.para_index

        for j in range(window_size - 1):
            # print self.word_dictionary[paragraph[self.word_index+j].lower()]
            # print Embedding.wordVec(paragraph[self.word_index+j].lower())

            # word_examples[i][j] = self.word_dictionary[paragraph[self.word_index+j].lower()]
            word_examples[i][j] = Embedding.wordIndex(paragraph[self.word_index+j].lower())
        # labels[i] = self.word_dictionary[paragraph[self.word_index+window_size-1].lower()]
        try:
            labels[i] = Embedding.wordIndex(paragraph[self.word_index+window_size-1].lower())
        except:
            import pdb; pdb.set_trace()
        self.word_index = self.word_index + 1

    return para_examples, word_examples, labels


class Phrase2Vec(object):

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.load_pretrained_word2vec()
        # print (self._session.run(self.wrd_emb))    
        self.build_graph()
        # print (self._session.run(self.wrd_emb))    

    def load_pretrained_word2vec(self):

        ckpt = tf.train.latest_checkpoint('save')
        saver = tf.train.import_meta_graph(ckpt+'.meta')
        saver.restore(self._session, ckpt)
        print("model restored.")

        all_var_names = [v.name for v in tf.global_variables()]
        assert "emb:0" in all_var_names
        self.wrd_emb = tf.get_default_graph().get_tensor_by_name("emb:0")
        wrd_emb = self._session.run(self.wrd_emb)
        self._options.vocabulary_size =  wrd_emb.shape[0]
        self._options.wrd_dim =  wrd_emb.shape[1]
        print("embedding loaded.")


    # load word2vec part with "trainable" false
    def train(self):

        batch_size = 5; window_size = 3

        for step in range(20000):
            phrases, words, labels = generate_batch(opts.batch_size, opts.window_size)

            # for i in range(phrases.size):
            #     print(phrases[i], words[i], '->', labels[i])

            # for i in range(phrase.size):
            #     print(phrases[i], reverse_phrases[batch[i]],
            #         words[i], reverse_dictionary[batch[i]],
            #         '->', labels[i], reverse_dictionary[labels[i]])

            feed_dict = {self.phr_examples:phrases, self.wrd_examples:words, self.labels:labels}
            import pdb; pdb.set_trace()
            _, loss_val = self._session.run([self.trainer, self.loss], feed_dict=feed_dict)

            if step % 100 == 0:
                print("loss at step ", step, ": ", loss_val)


    def build_graph(self):

        opts = self._options

        with tf.name_scope("phr"):

            # Input data
            self.phr_examples = tf.placeholder(tf.int32, shape=[batch_size], name="phr_examples")  
            self.wrd_examples = tf.placeholder(tf.int32, shape=[batch_size], name="wrd_examples")
            self.labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="labels")

            # In `concatenate` mode, total embedding is phr_dim + wrd_dim 
            emb_dim = opts.phr_dim + opts.wrd_dim
            
            # Phrase weights
            init_width = 0.5 / opts.phr_dim
            self.phr_emb = tf.Variable(tf.random_uniform([opts.phr_size, opts.phr_dim], -init_width, init_width), name="pr_emb")

            # Softmax weights (NCE)
            nce_weights = tf.Variable(tf.truncated_normal([opts.vocabulary_size, emb_dim], stddev=1.0 / math.sqrt(emb_dim)), name="nce_W")
            nce_biases = tf.Variable(tf.zeros([opts.vocabulary_size]), name="nce_b")

            # Global step: scalar, i.e., shape [].
            self.global_step = tf.Variable(0, name="global_step")
            
            # Variable initialize, and then, load word weights
            tf.variables_initializer([self.phr_emb, nce_weights, nce_biases, self.global_step], name='init').run()
            # tf.global_variables_initializer().run()
            
            # Embeddings for examples: [batch_size, emb_dim]
            example_phr_emb = tf.nn.embedding_lookup(self.phr_emb, self.phr_examples)
            example_wrd_emb = tf.nn.embedding_lookup(self.wrd_emb, self.wrd_examples)
            embed = tf.concat(1, [example_phr_emb, example_wrd_emb], name="combined")

            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                            biases=nce_biases,
                            labels=self.labels,
                            inputs=embed,
                            num_sampled=opts.num_sampled,
                            num_classes=opts.vocabulary_size), name="nceloss")

            # Construct the SGD optimizer using a learning rate of 1.0.
            # We only train phrase weights, softmax weights, and NOT word weights.
            optimizer = tf.train.GradientDescentOptimizer(1.0)
            trainer = optimizer.minimize(loss,
                                       global_step=self.global_step,
                                       gate_gradients=optimizer.GATE_NONE,
                                       var_list=[self.phr_emb, nce_weights, nce_biases])

        self.loss = loss
        self.trainer = trainer

        writer = tf.summary.FileWriter('save', graph=tf.get_default_graph())
        writer.close()


def main():

    """Train a word2vec model."""
    # if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    #     print("--train_data --eval_data and --save_path must be specified.")
    #     sys.exit(1)
    opts = Options()
    opts.phr_size = 100
    opts.phr_dim = 10
    opts.num_sampled = 64
    
    with tf.Graph().as_default(), tf.Session() as session:

        model = Phrase2Vec(opts, session)
        model.train()

    # # Perform a final save.
    # model.saver.save(session,
    #                  os.path.join(opts.save_path, "model.ckpt"),
    #                  global_step=model.global_step)

if __name__ == "__main__":
    main()
