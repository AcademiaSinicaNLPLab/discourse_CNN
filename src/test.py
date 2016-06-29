#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from keras.layers import Input, Lambda, merge
from keras import backend as K
from keras.models import Model
import numpy as np

def switch(xs):
    condition, then_expression, else_expression = xs
    if (K._BACKEND == 'tensorflow'):
        import tensorflow as tf
        return tf.select(tf.python.math_ops.greater(condition, K.zeros_like(condition)), then_expression, else_expression)
    else:
        return K.switch(indicator>0, then_expression, else_expression)

if __name__ == '__main__':
    input1 = Input((1,), dtype='int32')
    input2 = Input((2,), dtype='int32')
    input3 = Input((2,), dtype='int32')

    indicator = Lambda(lambda x:K.repeat_elements(x, 2, axis=1), output_shape=(2,))(input1)
    m = merge([indicator, input2, input3], mode=switch, output_shape=(None,2,))

    model = Model([input1, input2, input3], m)

    print model.predict([np.array([[0],[1]]), np.array([[0,1],[1,2]]), np.array([[3,4],[4,5]])])
