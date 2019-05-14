import tensorflow as tf
import numpy as np
from functools import partial

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print(X_train)
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
print(X_train)
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_index in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_index], y[batch_index]
        yield X_batch, y_batch


n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 20
n_outputs = 10

batch_norm_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()
    my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=batch_norm_momentum)
    my_dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)

    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))

    hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))

    hidden3 = my_dense_layer(bn2, n_hidden3, name="hidden3")
    bn3 = tf.nn.elu(my_batch_norm_layer(hidden3))

    hidden4 = my_dense_layer(bn3, n_hidden4, name="hidden4")
    bn4 = tf.nn.elu(my_batch_norm_layer(hidden4))

    logits_before_bn = my_dense_layer(bn4, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # training_op = optimizer.minimize(loss)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hiddden[4]|outputs")
    training_op = optimizer.minimize(loss, var_list=train_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]")
restore_saver = tf.train.Saver(reuse_vars)

n_epochs = 50
batch_size = 200

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./model_final.ckpt")
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        # print(bn1.eval(feed_dict={X: X_valid, y: y_valid}))
        print(epoch, "accuracy", accuracy_val)

    # save_path = saver.save(sess, "./model_final.ckpt")
    save_path = saver.save(sess, "./new_model_final.ckpt")

'''
saver = tf.train.import_meta_graph("./model_final.ckpt.meta")

for op in tf.get_default_graph().get_operations(): 
    print(op.name)

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")

training_op = tf.get_default_graph().get_operation_by_name("train/GradientDescent")

with tf.Session() as sess:
    saver.restore(sess, "./model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "accuracy", accuracy_val)

    save_path = saver.save(sess, "./new_model_final.ckpt")
'''