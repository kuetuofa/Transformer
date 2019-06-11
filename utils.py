# coding=utf-8
# ====================================
""" Utils.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

def transformer_flags():
    flags.DEFINE_string('dataset_path','/data/asadul/scribendiData/tsvFiles2019/data_small/',' Dataset Folder')
    flags.DEFINE_integer('buffer_size', 100000, 'Shuffle buffer size')
    flags.DEFINE_string('vocab_file','vocab.txt','Vocabulary file')
    flags.DEFINE_integer('sequence_length', 50, 'Maxinum number of words in a sequence')
    flags.DEFINE_integer('epochs', 10, 'Number of Epochs')
    flags.DEFINE_integer('batch_size', 256, 'Batch Size')
    flags.DEFINE_integer('per_replica_batch_size', 64, 'Batch Size')
    flags.DEFINE_integer('num_layers', 4, 'Nnmber of Encoder/Decoder Stack')
    flags.DEFINE_integer('d_model', 512, 'Output dimesion of all sublayers including Embedding layer')
    flags.DEFINE_integer('dff', 1024, 'Dimetionality of inner layer')
    flags.DEFINE_integer('num_heads', 4, 'Number of Attention Head')
    flags.DEFINE_boolean('enable_function', True, 'Enable Function')
    flags.DEFINE_integer('max_ckpt_keep', 5, 'Maximum Number of Checkpoint to keep')
    flags.DEFINE_string('ckpt_path', 'model_dist', 'Checkpoint Path')
    flags.DEFINE_float('dropout_rate', 0.1, 'Dropout Probability')


def flags_dict():
  """Define the flags.

  Returns:
    Command line arguments as Flags.
  """

  kwargs = {
      'dataset_path': FLAGS.dataset_path,
      'enable_function': FLAGS.enable_function,
      'buffer_size': FLAGS.buffer_size,
      'vocab_file': FLAGS.vocab_file,
      'batch_size': FLAGS.batch_size,
      'per_replica_batch_size': FLAGS.per_replica_batch_size,
      'sequence_length': FLAGS.sequence_length,
      'epochs': FLAGS.epochs,
      'num_layers': FLAGS.num_layers,
      'd_model': FLAGS.d_model,
      'dff': FLAGS.dff,
      'num_heads':FLAGS.num_heads,      
      'max_ckpt_keep': FLAGS.max_ckpt_keep,
      'ckpt_path': FLAGS.ckpt_path,
      'dropout_rate': FLAGS.dropout_rate,
  }

  return kwargs


def read_data(src_file, tgt_file):
    """Read text file and create tf.data

    Args:
        src_file: Source file.
        gt_file: Target file.
    
    Returns: 
        src_tgt_dataset: create tf.data as a zip source and target format

    """
    
    src_dataset=tf.data.TextLineDataset(tf.io.gfile.glob(src_file))
    tgt_dataset=tf.data.TextLineDataset(tf.io.gfile.glob(tgt_file))
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset,tgt_dataset))
    return src_tgt_dataset


def _load_or_create_tokenier(dataset, vocab_file):
    """Create tokeinizer if doen't exists

    Args:
        dataset: Training dataset to create the vocab file
        vocab_file: Path where to save vocab files
    
    Returns: 
        tokenizer: Tokenizer
    """

    if tf.io.gfile.exists(vocab_file+'.subwords')==False:
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (src.numpy() for src,tgt in dataset), target_vocab_size=2**15)
        tokenizer.save_to_file(vocab_file)
    else:
        print("Vocabulary exists Loading...")
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
            vocab_file)
    return tokenizer


def load_dataset(dataset_path, sequence_length, vocab_file, batch_size, buffer_size):
    """Create a tf.data Dataset.

    Args:
        dataset_path: Path to the files to load text from
        sequence_length: Maximun Length of the Sequence
        vocab_file: Location of vocab file
        batch_size: Batch size.
        buffer_size: Buffer size for suffling
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        tokenizer: input/output tokenizer
    """

    train_dataset = read_data(dataset_path+'train.src',
                            dataset_path+'train.tgt')
    test_dataset = read_data(dataset_path+'test.src', 
                            dataset_path+'test.tgt')

    tokenizer = _load_or_create_tokenier(train_dataset, vocab_file)
    
    def encode(src, tgt):
        src = [tokenizer.vocab_size] + tokenizer.encode(
            src.numpy()) + [tokenizer.vocab_size+1]

        tgt = [tokenizer.vocab_size] + tokenizer.encode(
            tgt.numpy()) + [tokenizer.vocab_size+1]
        return src, tgt

    def filter_max_length(x, y, max_length=sequence_length):
        return tf.logical_and(tf.size(x) <= max_length,
                            tf.size(y) <= max_length)

    def tf_encode(src, tgt):
        return tf.py_function(encode, [src, tgt], [tf.int64, tf.int64])
    
    train_dataset = train_dataset.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    train_dataset = train_dataset.shuffle(buffer_size).padded_batch(
        batch_size, padded_shapes=([-1],[-1]))

    test_dataset = test_dataset.map(tf_encode)
    test_dataset = test_dataset.padded_batch(
        batch_size, padded_shapes=([-1],[-1]))

    return train_dataset, test_dataset, tokenizer