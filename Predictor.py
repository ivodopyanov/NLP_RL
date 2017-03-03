import keras.backend as K
from keras.engine import Layer
from keras.activations import sigmoid, tanh, relu
from keras.initializations import glorot_uniform

import theano as T
import theano.tensor as TS
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import Print
from Base import Base

class Predictor(Base):
    def __init__(self, prob, **kwargs):
        self.prob = prob
        super(Predictor, self).__init__(**kwargs)


    def build(self, input_shape):
        super(Predictor, self).build(input_shape)
        self.stack = K.zeros((self.batch_size, self.max_len, 2*self.hidden_dim))
        self.cursors = K.concatenate([K.ones((self.batch_size, 1)), K.zeros((self.batch_size, self.max_len-1))], axis=1)
        self.stack_mask = K.zeros((self.batch_size, self.max_len))



    def compute_mask(self, input, input_mask=None):
        return [None, None, None, None, None, None]

    def get_output_shape_for(self, input_shape):
        return [(self.batch_size, self.hidden_dim),
                (self.batch_size, 2*self.max_len, self.hidden_dim),
                (self.batch_size, 2*self.max_len, self.hidden_dim),
                (self.batch_size, 2*self.max_len, self.hidden_dim),
                (self.batch_size, 2*self.max_len, 2),
                (self.batch_size, 2*self.max_len)]


    def call(self, input, mask=None):
        x = K.dot(input[0], self.W_emb) + self.b_emb
        bucket_size = input[1][0][0]


        initial_stack_current_value = K.zeros((self.batch_size, self.hidden_dim))
        initial_stack_prev_value = K.zeros((self.batch_size, self.hidden_dim))
        initial_input_current_value = K.zeros((self.batch_size, self.hidden_dim))
        initial_policy = K.zeros((self.batch_size, 2))
        initial_policy_calculated = K.zeros((self.batch_size,), dtype="int16")


        results, _ = T.scan(self.predictor_step,
                            outputs_info=[self.stack, self.cursors, self.stack_mask,
                                          initial_stack_current_value,
                                          initial_stack_prev_value,
                                          initial_input_current_value,
                                          initial_policy,
                                          initial_policy_calculated],
                            non_sequences=[x, mask[0]],
                            n_steps=2*bucket_size)
        last_value = results[0][-1]
        stack_current_values = results[3]
        stack_prev_values = results[4]
        input_current_values = results[5]
        policy_values = results[6]
        policy_calculated = results[7]

        stack_current_values = stack_current_values.dimshuffle([1,0,2])
        stack_prev_values = stack_prev_values.dimshuffle([1,0,2])
        input_current_values = input_current_values.dimshuffle([1,0,2])
        policy_values = policy_values.dimshuffle([1,0,2])
        policy_calculated = policy_calculated.dimshuffle([1,0])

        return [last_value[:,0,self.hidden_dim:], stack_current_values, stack_prev_values, input_current_values, policy_values, policy_calculated]


    #1 = SHIFT
    #0 = REDUCE
    def predictor_step(self, stack, cursors, stack_mask, stack_current_value_tm1, stack_prev_value_tm1, input_current_value_tm1, policy_tm1, policy_calculated_tm1, data, mask):
        value_masks = self.get_value_masks(cursors, stack_mask)
        values = self.get_values(stack, data, value_masks)

        reduced = self.calc_reduced_value(values)
        action, policy = self.calc_action(values)

        '''rng = RandomStreams()
        random_action_mask = rng.choice(size=(self.batch_size,), p=[1-self.prob, self.prob])
        random_actions = rng.choice(size=(self.batch_size,))
        action = action*(1-random_action_mask) + random_actions*random_action_mask'''

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

        stack_current_value = values['stack_current'][:, self.hidden_dim:]
        stack_prev_value = values['stack_prev'][:, self.hidden_dim:]
        input_current_value = values['input_current'][:, self.hidden_dim:]

        return new_stack, new_cursors, new_stack_mask, stack_current_value, stack_prev_value, input_current_value, policy, policy_calculated