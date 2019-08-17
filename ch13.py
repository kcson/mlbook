import numpy as np
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")


def plot_color_image(image):
    plt.imshow(image.astype(np.uint8), interpolation="nearest")
    plt.axis("off")


china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape
print(china.shape)
print(dataset.shape)

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
# filter
filters[:, 3, :, 0] = 1  # 수직
filters[3, :, :, 1] = 1  # 수평

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")
convolution = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2, 2], padding="SAME")
max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output = sess.run(convolution, feed_dict={X: dataset})
    pooling = sess.run(max_pool, feed_dict={X: dataset})

print(pooling.shape)
# plt.imshow(output[1, :, :, 1], cmap="gray")
# plt.show()
plot_color_image(pooling[0])
plt.show()
plot_color_image(dataset[0])
plt.show()

for image_index in (0, 1):
    for feature_map_index in (0, 1):
        plt.imshow(output[image_index, :, :, feature_map_index], cmap="gray")
        plt.show()
