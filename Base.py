import keras.backend as K
from keras.engine import Layer
from keras.activations import sigmoid, tanh, relu
from keras.initializations import glorot_uniform

import theano as T
import theano.tensor as TS
from theano.printing import Print

class Base(Layer):
    def __init__(self, input_dim, hidden_dim, RL_dim, max_len, batch_size, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.RL_dim = RL_dim
        self.max_len = max_len
        self.batch_size = batch_size
        super(Base, self).__init__(**kwargs)


    def build(self, input_shape):
        self.W_emb = glorot_uniform((self.input_dim, 2*self.hidden_dim), name='{}_W_emb'.format(self.name))
        self.b_emb = K.zeros((2*self.hidden_dim,), name='{}_b_emb'.format(self.name))

        self.W_R = glorot_uniform((2*self.hidden_dim, 5*self.hidden_dim), name='{}_W_R'.format(self.name))
        self.b_R = K.zeros((5*self.hidden_dim,), name='{}_b_R'.format(self.name))

        self.W_S1 = glorot_uniform((3*self.hidden_dim, self.RL_dim), name='{}_W_S1'.format(self.name))
        self.b_S1 = K.zeros((self.RL_dim,), name='{}_b_S1'.format(self.name))
        self.W_S2 = glorot_uniform((self.RL_dim,2), name='{}_W_S2'.format(self.name))
        self.b_S2 = K.zeros((2,), name='{}_b_S2'.format(self.name))

        self.built = True

    def get_value_masks(self, cursors, stack_mask):
        stack_mask_prev_prev_value = self.get_prev_prev_value_mask(stack_mask)
        stack_mask_prev_value = self.get_prev_value_mask(stack_mask)
        stack_mask_current_value = self.get_current_value_mask(stack_mask)
        stack_mask_next_value = self.get_next_value_mask(stack_mask)
        cursor_next_value = self.get_next_value_mask(cursors)
        cursor_current_value = self.get_current_value_mask(cursors)
        return {'stack_prev_prev': stack_mask_prev_prev_value,
                'stack_prev': stack_mask_prev_value,
                'stack_current': stack_mask_current_value,
                'stack_next': stack_mask_next_value,
                'input_current': cursor_current_value,
                'input_next': cursor_next_value}

    def get_values(self, stack, data, value_masks):
        # Выделяем текущее значение во входных данных (на которое указывают cursors)
        input_current_value = self.get_value_by_bitmask(data, value_masks['input_current'])
        # Выделяем два последних значения из стэка
        stack_prev_value = self.get_value_by_bitmask(stack, value_masks['stack_prev'])
        stack_prev_prev_value = self.get_value_by_bitmask(stack, value_masks['stack_prev_prev'])

        return {'input_current': input_current_value,
                'stack_prev': stack_prev_value,
                'stack_prev_prev': stack_prev_prev_value}

    def calc_reduced_value(self, values):
        # Вычисляем новое значение для операции REDUCE, полученное из двух последних векторов из стека
        h = K.concatenate([values['stack_prev'][:, self.hidden_dim:], values['stack_prev_prev'][:, self.hidden_dim:]], axis=1)
        q = K.dot(h, self.W_R)+self.b_R
        q1 = sigmoid(q[:, :4*self.hidden_dim])
        q2 = tanh(q[:, 4*self.hidden_dim:])
        c = q1[:, self.hidden_dim:2*self.hidden_dim]*values['stack_prev'][:,:self.hidden_dim] + \
            q1[:, 2*self.hidden_dim:3*self.hidden_dim]*values['stack_prev_prev'][:,:self.hidden_dim] + \
            q1[:, :self.hidden_dim]*q2
        h = q1[:, 3*self.hidden_dim:]*c
        reduced = K.concatenate([c, h], axis=1)
        return reduced

    def calc_action(self, values):
        # Вычисляем действие, которое надо выполнить
        s_input = K.concatenate([values['stack_prev'][:, self.hidden_dim:],
                                 values['stack_prev_prev'][:, self.hidden_dim:],
                                 values['input_current'][:, self.hidden_dim:]], axis=1)
        s = relu(K.dot(s_input, self.W_S1)+self.b_S1)
        policy = K.exp(K.dot(s, self.W_S2)+self.b_S2)
        action = TS.switch(TS.ge(policy[:,0], policy[:,1]), 1, 0)
        return action, policy

    def apply_border_conditions(self, cursors, stack_mask, mask, action):
        # Обрабатываем случаи, когда у агента есть только один выбор (закончились входные данные или в стеке нет двух элементов)
        # Или вообще не надо делать выбор (вся строка посчитана)
        sentence_length = K.sum(mask, axis=1)
        cursor_pos = K.sum(cursors, axis=1)
        stack_size = K.sum(stack_mask, axis=1)

        input_is_empty = TS.eq(sentence_length-cursor_pos, -1)
        stack_is_empty = TS.le(stack_size, 1)
        action = TS.switch(input_is_empty, 0, action)
        action = TS.switch(stack_is_empty, 1, action)
        no_action = TS.and_(input_is_empty, stack_is_empty)
        policy_calculated = 1-TS.or_(input_is_empty, stack_is_empty)
        #action = Print("action")(action)
        #no_action = Print("no_action")(no_action)
        return action, no_action, policy_calculated

    def calc_stack_after_reduce(self, stack, value_masks, reduced):
        # Формируем новое состояние стека для операции REDUCE
        # Обнуляем последний элемент стека и вставляем в предпоследний - новое значение
        stack_reduce_result = self.insert_tensor_at_mask(stack, value_masks['stack_prev'], K.zeros((self.batch_size, 2*self.hidden_dim)))
        stack_reduce_result = self.insert_tensor_at_mask(stack_reduce_result, value_masks['stack_prev_prev'], reduced)
        return stack_reduce_result

    def calc_stack_after_shift(self, stack, value_masks, values):
        # Формируем новое состояние стека для операции SHIFT
        # Просто добавляем текущий элемент из input
        stack_shift_result = self.insert_tensor_at_mask(stack, value_masks['stack_next'], values['input_current'])
        return stack_shift_result





    #Из маски 11111000000 делает маску 0000100000 (1 - на последнем элементе 1 исходной маски, остальное - 0)
    def get_current_value_mask(self, mask):
        result = 1 - mask
        result = K.concatenate([result[:, 1:], K.ones((self.batch_size, 1))], axis=1)
        result = mask*result
        return result

    def get_next_value_mask(self, mask):
        result = K.concatenate([K.ones((self.batch_size, 1)), mask[:,:-1]], axis=1)
        return self.get_current_value_mask(result)

    def get_prev_value_mask(self, mask):
        result = K.concatenate([mask[:, 1:], K.zeros((self.batch_size, 1))], axis=1)
        return self.get_current_value_mask(result)

    def get_prev_prev_value_mask(self, mask):
        result = K.concatenate([mask[:, 2:], K.zeros((self.batch_size, 2))], axis=1)
        return self.get_current_value_mask(result)

    def get_value_by_bitmask(self, data, mask):
        mask = mask.dimshuffle([0,1,'x'])
        mask = TS.extra_ops.repeat(mask, 2*self.hidden_dim, axis=2)
        value = K.switch(mask, data, 0)
        value = K.sum(value, axis=1)
        return value

    def insert_tensor_at_mask(self, data, mask, tensor):
        # Получаем матрицу из тензора (который надо вставить), скопированных max_len раз
        tensor = K.repeat(tensor, self.max_len)
        # Получаем маску, куда надо вставить этот тензор
        mask = mask.dimshuffle([0,1,'x'])
        mask = TS.extra_ops.repeat(mask, 2*self.hidden_dim, axis=2)
        # Вставляем, используя switch
        result = K.switch(mask, tensor, data)
        return result