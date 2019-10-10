import tensorflow as tf
from Encoder import Encoder
from Decoder import Decoder
from positionalencoding import create_masks
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
  
  #@tf.function(input_signature=[tf.TensorSpec([None,None],tf.int64,name='inp'),
  #                              tf.TensorSpec([None,None],tf.int64,name='out'),
  #                              tf.TensorSpec(None,tf.bool,name='flag')])  
  def call(self, input_, training=False):
    inp, tar = input_
    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp,tar)
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights
