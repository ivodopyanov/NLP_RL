import keras.backend as K
from keras.engine import Layer
from keras.activations import sigmoid, tanh, relu
from keras.initializations import glorot_uniform

import theano as T
import theano.tensor as TS
from theano.printing import Print
from Base import Base

class Encoder(Base):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)


    def build(self, input_shape):
        super(Encoder, self).build(input_shape)
        self.trainable_weights = [self.W_emb, self.b_emb, self.W_R, self.b_R]



    def compute_mask(self, input, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (self.batch_size, self.hidden_dim)


    def call(self, input, mask=None):
        x = K.dot(input[0], self.W_emb) + self.b_emb
        bucket_size = input[1][0][0]

        stack = K.zeros((self.batch_size, self.max_len, 2*self.hidden_dim))
        cursors = K.concatenate([K.ones((self.batch_size, 1)), K.zeros((self.batch_size, self.max_len-1))], axis=1)
        stack_mask = K.zeros((self.batch_size, self.max_len))

        results, _ = T.scan(self.encoder_step,
                            outputs_info=[stack, cursors, stack_mask],
                            non_sequences=[x, mask[0]],
                            n_steps=2*bucket_size)
        last_value = results[0][-1]
        return last_value[:,0,self.hidden_dim:]



    #1 = SHIFT
    #0 = REDUCE
    def encoder_step(self, stack, cursors, stack_mask, data, mask):
        value_masks = self.get_value_masks(cursors, stack_mask)
        values = self.get_values(stack, data, value_masks)

        reduced = self.calc_reduced_value(values)
        action, policy = self.calc_action(values)
        action, no_action, policy_calculated = self.apply_border_conditions(cursors, stack_mask, mask, action)
        stack_reduce_result = self.calc_stack_after_reduce(stack, value_masks, reduced)
        stack_shift_result = self.calc_stack_after_shift(stack, value_masks, values)

        # Итоговое новое состояние агента в зависимости от выполняемых операций (SHIFT или REDUCE)
        do_shift = TS.eq(action, 1)
        do_shift = do_shift.dimshuffle([0,'x'])
        do_shift = TS.extra_ops.repeat(do_shift, self.max_len, axis=1)
        new_cursors = K.switch(do_shift, cursors + value_masks['input_next'], cursors)
        new_stack_mask = K.switch(do_shift, stack_mask + value_masks['stack_next'], stack_mask - value_masks['stack_current'])
        do_shift = do_shift.dimshuffle([0, 1,'x'])
        do_shift = TS.extra_ops.repeat(do_shift, 2*self.hidden_dim, axis=2)
        new_stack = K.switch(do_shift, stack_shift_result, stack_reduce_result)

        no_action_mask = no_action.dimshuffle([0,'x'])
        no_action_mask = TS.extra_ops.repeat(no_action_mask, self.max_len, axis=1)
        new_cursors = K.switch(no_action_mask, cursors, new_cursors)
        new_stack_mask = K.switch(no_action_mask, stack_mask, new_stack_mask)
        no_action_mask = no_action_mask.dimshuffle([0, 1,'x'])
        no_action_mask = TS.extra_ops.repeat(no_action_mask, 2*self.hidden_dim, axis=2)
        new_stack = K.switch(no_action_mask, stack, new_stack)

        return new_stack, new_cursors, new_stack_mask