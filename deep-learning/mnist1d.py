import tensorflow as tf
import numpy as np
import sys
import pickle

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

# MNIST-1D dataset from "https://github.com/greydanus/mnist1d"

# 0. data processing
dataset = from_pickle('mnist1d_data.pkl')

x_train = dataset['x']
x_test = dataset['x_test']
y_train = dataset['y']
y_test = dataset['y_test']

train_data_num = len(x_train)
test_data_num = len(x_test)
tmp_x_train = [x_train[i].reshape(1, -1) for i in range(train_data_num)]
tmp_x_test = [x_test[i].reshape(1, -1) for i in range(test_data_num)]
x_train = tmp_x_train
x_test = tmp_x_test

y_train = y_train[:train_data_num]
y_train = np.array(y_train).reshape(-1)
y_train = np.eye(10)[y_train]
y_train = y_train.reshape(len(y_train), 1, 10)

y_test = y_test[:test_data_num]
y_test = np.array(y_test).reshape(-1)
y_test = np.eye(10)[y_test]
y_test = y_test.reshape(len(y_test), 1, 10)

print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))
print(np.shape(y_test))

# 1. training
tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder(tf.float32, [1, 40])  # batch=1, hard-coded, 40
Y = tf.compat.v1.placeholder(tf.float32, [1, 10])   # 10 labels (0~9)

W1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([40, 16], stddev=0.01)) # first layer
L1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(X, W1))

W2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 10], stddev=0.01)) # last layer
model = tf.compat.v1.matmul(L1, W2)

cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cost)

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

batch_size = 1
total_batch = int(len(x_train) / batch_size)

for epoch in range(3):
    total_cost = 0
    for i in range(total_batch):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_train[i], Y: y_train[i]})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

# 2-1. inference by tensorflow (with no weight quantization)
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
success = 0
for i in range(test_data_num):
    res = sess.run(is_correct, feed_dict={X: x_test[i], Y: y_test[i]})
    if res[0] == True: success += 1
print("success:", success)

# 2-2. inference by tensorflow (with weight quantization)
# weight quantization => input[i] = trunc(input[i] * 1000), int32, non-negative value
# input[i] = trunc(input[i] * 1000)
scale = 100

for x in x_train:
    for i in range(len(x)):
        x[i] = np.trunc(x[i] * scale)
for x in x_test:
    for i in range(len(x)):
        x[i] = np.trunc(x[i] * scale)

x_train = np.array(x_train, np.int32)
y_train = np.array(y_train, np.int32)
x_test = np.array(x_test, np.int32)
y_test = np.array(y_test, np.int32)

def weight_quantization(w, mul):
    wn = sess.run(w)
    for i in range(wn.shape[0]):
        wni = wn[i]
        for j in range(wn.shape[1]):
            wni[j] = np.trunc(wni[j] * mul)
            #wni[j] += scale_add
    return wn

w1n = weight_quantization(W1, scale)
w2n = weight_quantization(W2, scale)

W1 = tf.compat.v1.assign(W1, w1n)
W2 = tf.compat.v1.assign(W2, w2n)

L1_SCALE = tf.compat.v1.nn.relu(tf.compat.v1.matmul(X, W1))
model_scale = tf.compat.v1.matmul(L1_SCALE, W2)
argmax_model = tf.argmax(model_scale, 1)
is_correct = tf.equal(argmax_model, tf.argmax(Y, 1))

success = 0
for i in range(test_data_num):
    m, max, res = sess.run([model_scale, argmax_model, is_correct], feed_dict={X: x_test[i], Y: y_test[i]})
    if i>=0 and i<10:
        print(m)

    if res[0] == True: success += 1
print("success:", success)

# 3. dump model parameter (as csv file)
def dump_param(w, filename, sess):
    with open(filename, 'w') as f:
        sys.stdout = f
        wn = sess.run(w)
        for i in range(wn.shape[0]):
            wni = wn[i]
            line_str = ""
            for j in range(wn.shape[1]):
                line_str += (str(int(wni[j])) + ",")
            print(line_str)

dump_param(W1, 'w1_1d_40_16.csv', sess)
dump_param(W2, 'w2_1d_16_10.csv', sess)

# 4. dump test data
with open('x_test.csv', 'w') as f:
    sys.stdout = f
    for i in range(10):
        line_str = ""
        x = x_test[i][0]
        for j in range(x_test.shape[2]):
            line_str += (str(int(x[j])) + ",")
        print(line_str)

with open('y_test.csv', 'w') as f:
    sys.stdout = f
    for i in range(10):
        res = np.argmax(y_test[i])
        print(res)
