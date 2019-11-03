import numpy as np
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# 그림을 저장할 폴더
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


n_inputs = 3
n_neurons = 5

###############################################################################################
# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
# Y0, Y1 = output_seqs
# Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
# Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
# b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
#
# Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
# Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

# init = tf.global_variables_initializer()
#
# X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])  # t = 0
# X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])  # t = 1
#
# with tf.Session() as sess:
#     init.run()
#     Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
#
# print(Y0_val)
# print(Y1_val)

########################################################################################
n_steps = 3
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
# outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])
X_batch = np.array([
    # t = 0      t = 1       t = 2
    [[0, 1, 2], [9, 8, 7], [9, 8, 7]],  # 샘플 1
    [[3, 4, 5], [0, 0, 0], [9, 8, 7]],  # 샘플 2
    [[6, 7, 8], [6, 5, 4], [0, 0, 0]],  # 샘플 3
    [[9, 0, 1], [3, 2, 1], [9, 8, 7]],  # 샘플 4
    [[9, 0, 1], [3, 2, 1], [9, 8, 7]]  # 샘플 5
])
seq_length_batch = np.array([3, 3, 2, 3, 3])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    # output_val = outputs.eval(feed_dict={X: X_batch})
    output_val, state_val = sess.run([outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

print(output_val)
print(state_val)
