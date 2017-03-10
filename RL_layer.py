import keras.backend as K
from keras.engine import Layer
from keras.activations import sigmoid, tanh, relu
from keras.initializations import glorot_uniform

import theano as T
import theano.tensor as TS

class RL_layer(Layer):
    def __init__(self, hidden_dim, RL_dim, **kwargs):
        self.hidden_dim = hidden_dim
        self.RL_dim = RL_dim
        super(RL_layer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.W_S1 = glorot_uniform((3*self.hidden_dim, self.RL_dim), name='{}_W_S1'.format(self.name))
        self.b_S1 = K.zeros((self.RL_dim,), name='{}_b_S1'.format(self.name))
        self.W_S2 = glorot_uniform((self.RL_dim,2), name='{}_W_S2'.format(self.name))
        self.b_S2 = K.zeros((2,), name='{}_b_S2'.format(self.name))

        self.trainable_weights = [self.W_S1, self.b_S1, self.W_S2, self.b_S2]
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 2)


    def call(self, input, mask=None):
        prev_values = input[0]
        prev_prev_values = input[1]
        current_values = input[2]


        s_input = K.concatenate([prev_values, prev_prev_values, current_values], axis=1)
        s = relu(K.dot(s_input, self.W_S1)+self.b_S1)
        policy = -K.exp(K.dot(s, self.W_S2)+self.b_S2)
        return policy
