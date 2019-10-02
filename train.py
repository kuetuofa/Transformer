from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import time
import sys
import pdb 
import datetime 
from absl import flags
from absl import app

import tensorflow as tf

import utils
from Transformer import Transformer

assert tf.__version__.startswith('2')

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=60000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Train(object):
    """ Train class

    Args:
        epochs: Number of Epochs
        enable_function: Decorate function with tf.function
        transformer: Transformer
        src_tokenizer: Source Tokenizer
        tgt_tokenizer: target Tokenizer
        batch_size: Batch size
        train_log_dir: Training log directory
        test_log_dir: Test Log directory
        max_ckpt_keep: Maximum Number of Checkpoint to keep
        ckpt_path: Checkpoint path
        d_model: Output dimesion of all sublayers including Embedding layer 

    """

    def __init__(self, epochs, enable_function, transformer, src_tokenizer, tgt_tokenizer,
                batch_size,  train_log_dir, test_log_dir,
                max_ckpt_keep, ckpt_path, d_model):
        self.epochs = epochs
        self.enable_function = enable_function
        self.transformer = transformer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.batch_size = batch_size
        self.learning_rate =  CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        # reduction=tf.keras.losses.Reduction.SUM)
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                           optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_path, 
                            max_to_keep=max_ckpt_keep)


    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) * 1./self.batch_size

    
    @tf.function  
    def predict(self,input_sentence):
        """ Greedy Inference

        Args:
            input sentence: Input Sentence
        Return:
            result: predicted result of the input sentence
        """
        start_token =[self.src_tokenizer.vocab_size]
        end_token =[self.src_tokenizer.vocab_size+1]
        print(input_sentence)
        input_sentence= input_sentence[0]
        input_sentence = start_token + self.src_tokenizer.encode(input_sentence) + end_token
        encoder_input = tf.cast(tf.expand_dims(input_sentence,0),tf.int64)
        output = tf.cast(tf.expand_dims(start_token, 0),tf.int64)
        result=''

        for  i in range(10):
            predictions, _=self.transformer.call(encoder_input,output,False)
            predictions=predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int64)

            #if tf.equal(predicted_id, self.tokenizer.vocab_size+1):
            #        result=tf.squeeze(output, axis=0)
            #        break

            output = tf.concat([output, predicted_id], axis=-1)
        if result=='':
            result=tf.squeeze(output, axis=0)
        return result


    def train_step(self, inputs):
        """ One Training Step
        Args:
            inputs: Tuple of input tensor, target tensor
        """
        src, tar = inputs
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
            
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(src, tar_inp,training=True )
            loss = self.loss_function(tar_real, predictions)
            
        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
    
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)


    def test_step(self, inputs):
        """ One Test Step
        Args:
            inputs: Tuple of input tensor, target tensor
        """
        src, tar = inputs
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        predictions, _=self.transformer(src, tar_inp, training=False)

        t_loss = self.loss_function(tar_real, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(tar_real,predictions)
    
    
    def load_ckpt(self):
        """if a checkpoint exists, restore the latest checkpoint.
        """
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            print ('Latest checkpoint restored!!')

    
    def training_loop(self, train_dataset, test_dataset):
        """Custom training and testing loop.

        Args:
            train_dataset: Training dataset
            test_dataset: Testing dataset
        """

        if  self.enable_function:
            self.train_step = tf.function(self.train_step)
            self.test_step = tf.function(self.test_step)
        template = 'Epoch {}  Loss {:.4f} Accuracy {:.4f}, Test Loss {:.4f}, Test Accuracy {:.4f}'

        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()

            start = time.time()
            counter = 0

            for src,tgt in train_dataset:
                #print(counter)
                self.train_step((src,tgt))
                counter +=1
                if (counter+1)%100==0:
                    for t_src,t_tgt in test_dataset:
                        self.test_step((t_src,t_tgt))
                    print (template.format(
                        epoch + 1, self.train_loss.result(), self.train_accuracy.result()*100,
                        self.test_loss.result(), self.test_accuracy.result()*100))
               
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)

            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)
            
            if (epoch + 1) % 5 or (epoch+1==self.epochs) == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

            print ('Time taken for {} epoch: {} secs\n'.format(epoch+1, (time.time() - start)))
    

def run_main(argv):
    del argv
    kwargs= utils.flags_dict()
    main(**kwargs)


def main(epochs, enable_function, buffer_size, batch_size, d_model, dff, num_heads, vocab_file,
        dataset_path,dropout_rate, num_layers,sequence_length,ckpt_path, max_ckpt_keep):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    train_dataset, test_dataset, src_tokenizer, tgt_tokenizer = utils.load_dataset(dataset_path, 
                                sequence_length,vocab_file, batch_size, buffer_size)
    input_vocab_size = src_tokenizer.vocab_size + 2
    target_vocab_size = tgt_tokenizer.vocab_size + 2
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)
    
    print('create training object')
    train_obj=Train(epochs, enable_function, transformer, src_tokenizer,
            tgt_tokenizer, batch_size, train_log_dir, test_log_dir,
            max_ckpt_keep, ckpt_path,d_model)
    
    train_obj.training_loop(train_dataset,test_dataset)
    train_obj.load_ckpt()
    input_sentence='he go to school'
    train_obj.predict(input_sentence)
    tf.saved_model.save(train_obj.transformer, 'model')


if __name__=='__main__':
    utils.transformer_flags()
    app.run(run_main)