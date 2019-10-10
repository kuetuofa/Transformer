from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import sys
import pdb 
import datetime 
from absl import flags
from absl import app

import tensorflow as tf

import utils
from Transformer import Transformer
from train import Train

assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_gpu', 4, 'Number of GPUs to use')

class DistributedTrain(Train):
    def __init__(self, epochs, enable_function, transformer, src_tokenizer, tgt_tokenizer,
                batch_size, per_replica_batch_size, train_log_dir, test_log_dir,
                max_ckpt_keep, ckpt_path,d_model):
        """ Train class

        Args:
            epochs: Number of Epochs
            enable_function: Decorate function with tf.function
            transformer: Transformer
            src_tokenizer: source tokenizer
            tgt_tokenizer: target tokenizer
            batch_size: Batch size
            train_log_dir: Training log directory
            test_log_dir: Test Log directory
            max_ckpt_keep: Maximum Number of Checkpoint to keep
            ckpt_path: Checkpoint path
            d_model: Output dimesion of all sublayers including Embedding layer 

        """
        Train.__init__(self, epochs, enable_function, transformer, src_tokenizer,
                tgt_tokenizer, batch_size,  train_log_dir, test_log_dir,
                max_ckpt_keep, ckpt_path,d_model)
    
   
    def training_loop(self, train_iterator, test_iterator, strategy):
        """Custom training and testing loop.

        Args:
            train_dataset: Training dataset
            test_dataset: Testing dataset
            strategy: Distribution strategy
        """
        def distributed_train():
            return strategy.experimental_run(self.train_step, train_iterator)

        def distributed_test():
            return strategy.experimental_run(self.test_step, test_iterator)
        
        if  self.enable_function:
            distributed_train = tf.function(distributed_train)
            distributed_test = tf.function(distributed_test)

        template = 'Epoch {} Steps {} Train Loss {:.4f} Accuracy {:.4f}, Valid Loss {:.4f}, Test Accuracy {:.4f} Times takes {: .4f} seconds'

        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()

            start_time = time.time()
            train_steps = 0
            test_steps = 0
            train_iterator.initialize()
            while True:
                step_timer = time.time()
                try:
                    distributed_train()
                    train_steps +=1
                    if train_steps %500==0:
                        test_iterator.initialize()
                        
                        while True:
                            try:
                                test_steps +=1
                                distributed_test()
                            except tf.errors.OutOfRangeError:
                                break
                            if test_steps% 100:
                                break

                        print (template.format(
                        epoch + 1, train_steps, self.train_loss.result(), self.train_accuracy.result()*100,
                        self.test_loss.result(), self.test_accuracy.result()*100, time.time()-step_timer))
                except tf.errors.OutOfRangeError:
                    break

            
            print (template.format(
                        epoch + 1, train_steps, self.train_loss.result(), self.train_accuracy.result()*100,
                        self.test_loss.result(), self.test_accuracy.result()*100, time.time()-step_timer))

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)
            
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)

            if (epoch + 1) % 5 or ((epoch+1)==self.epochs)== 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
        
            print ('Time taken for {} epoch: {} secs\n'.format(epoch+1, (time.time() - start_time)))
    
     
def run_main(argv):
    del argv
    kwargs= utils.flags_dict()
    kwargs.update({'num_gpu': FLAGS.num_gpu})
    main(**kwargs)


def main(epochs, enable_function, buffer_size, batch_size, d_model, dff, num_heads, src_vocab_file, tgt_vocab_file,
        dataset_path,dropout_rate, num_layers,sequence_length, per_replica_batch_size,ckpt_path, max_ckpt_keep, num_gpu=1):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    devices =['/device:GPU:{}'.format(i) for i in range(num_gpu)]
    strategy = tf.distribute.MirroredStrategy(devices)
    num_replicas = strategy.num_replicas_in_sync
    
    with strategy.scope():
        train_dataset, test_dataset, src_tokenizer, tgt_tokenizer = utils.load_dataset(dataset_path, 
                                    sequence_length,src_vocab_file, tgt_vocab_file, batch_size, buffer_size)
        input_vocab_size  = src_tokenizer.vocab_size + 2
        target_vocab_size = tgt_tokenizer.vocab_size + 2
        #pdb.set_trace()        
        #num_train_steps_per_epoch = len(list(train_dataset))#tf.data.experimental.cardinality(train_dataset)
        #num_test_steps_per_epoch = len(list(test_dataset))#tf.data.experimental.cardinality(test_dataset)
        #print('Num Train Steps',num_train_steps_per_epoch)

        train_iterator = strategy.make_dataset_iterator(train_dataset)
        test_iterator = strategy.make_dataset_iterator(test_dataset)

        local_batch_size, remainder = divmod(batch_size, num_replicas)
        template = ('Batch size ({}) must be divisible by the '
                'number of replicas ({})')
        if remainder:
            raise ValueError(template.format(batch_size, num_replicas))

        transformer = Transformer(num_layers, d_model, num_heads, dff,
                            input_vocab_size, target_vocab_size, dropout_rate)
        
        print('create training object')
        train_obj=DistributedTrain(epochs, enable_function, transformer, src_tokenizer,
            tgt_tokenizer, batch_size, local_batch_size, train_log_dir, test_log_dir,
            max_ckpt_keep, ckpt_path,d_model)
        print('Training.....')
        
        train_obj.training_loop(train_iterator,
                                test_iterator,
                                strategy )
       # train_obj.load_ckpt()
        print(train_obj.predict(['he goes to school']))
        tf.saved_model.save(train_obj.transformer, 'model')#,signatures=call)


if __name__=='__main__':
    utils.transformer_flags()
    app.run(run_main)
