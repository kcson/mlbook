import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from datetime import datetime

# x = tf.Variable(3, name="x")
# y = tf.Variable(4, name="y")
# f = x * x * y + y + 2

# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
# print(result)
# sess.close()
# init = tf.global_variables_initializer()
#
# sess = tf.InteractiveSession()
#
# init.run()
# result = f.eval()
# print(result)
#
# sess.close()
#
# x1 = tf.Variable(1)
# print(x1.graph is tf.get_default_graph())
#########################################################################

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
n_epochs = 10
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
scaler.fit(housing.data)
scaled_housing_data = scaler.transform(housing.data)

scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# print(scaled_housing_data_plus_bias)

# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1, 1, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error))

# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

mse_summary = tf.summary.scalar("MSE", mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        # if epoch % 100 == 0:
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            # print("Epoch", epoch, "MSE=", mse.eval())

    print(theta.eval())
    save_path = saver.save(sess, "C:\\Users\\fob33\\mywork\\PythonProject\\mlbook\\model_final.ckpt")
    file_writer.close()
