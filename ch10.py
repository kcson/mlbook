import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
#
# config = tf.contrib.learn.RunConfig(tf_random_seed=42)
# feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_cols, config=config)
# dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
#
# tf.logging.set_verbosity(tf.logging.INFO)
# dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)
#
# y_pred = dnn_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred['classes']))
######################################################################################
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros[n_neurons], name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 40
batch_size = 50


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#         print(epoch, "batch : ", acc_batch, "valid : ", acc_valid)
#
#     save_path = saver.save(sess, "./model_final.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "./model_final.ckpt")
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

print("target : ", y_test[:20])
print("pred  : ", y_pred)
