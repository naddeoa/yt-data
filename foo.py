import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # use cpu because I'm training on gpu


import tensorflow as tf
from keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, LeakyReLU

a = tf.constant(
    [[
        [0,0],
        [1,1]
    ]]
)


print()
print(a)
print(a.shape)

print()
print('flatten')
f = Flatten(input_shape=(a.shape))(a)
print(f.shape)
print(f)

print('flatten and revert')
r = Reshape((2,2))(f)
print(r)
print(r.shape)