import tensorflow as tf
import matplotlib.pyplot as plt

filename = os.path.join("C:\\Users\\longyiyuan\\Downloads\\train\\train", )
def read_my_file_fromat(finename_queue):
	reader = tf.SomeReader()
	key, record_string = reader.read(filename_queue)
	example, label = tf.some_decoder(record_string)
	processed_example = some_processing(example)
	return processed_example, label

def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames,
		num_epochs=num_epochs, shuffle=True)
	example_list = [read_my_file_format(filename_queue) for _ in range(read_threads)]
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch_join(
		example_list, batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)
	return example_batch, label_batch

#define the placeholder of the network
xs = tf.placeholder(tf.float32, [None,1024])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 32, 32, 3])
#define the weight_variable funciton
def Weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


#define the bias_variable funtion
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#define the conv funciton 
def  conv2d(inputs, weight):
	return tf.nn.con2d(inputs, weight, strides=[1,1,1,1], padding='SAME')

#define the pool fuction
def max_pooling_2x2(inputs):
	return tf.nn.max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#define the computer accracy function
def computer_accuracy(v_xs, v_ys):
	global prediction
	y_re = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	current_prediction = tf.equal(tf.argmax(y_re, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(current_prediciton, tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
	return result

#the first convolution
W_conv1 = Weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W.conv1) + b_conv1)
h_pool1 = max_pooling_2x2(h_conv1)

#the second convolution
W_conv2 = Weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_con2) + b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)

#the third convolution
W_conv3 = Weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv2) + b_conv3)
h_pool3 = max_pooling_2x2(h_conv3)

#the first full connection layer
w_f1 = Weight_variable(8*8*128, 1024)
b_f1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool3, [-1, 8*8*128])
h_f1_drop = tf.nn.drop(h_f1, keep_prob)

w_f2 = Weight_variable([1024, 2])
b_f2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matul(h_f1_drop, w_f2) + b_f2)

#the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(-ys*tf.log(prediction), 
	reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimiaze(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

y_data = {}
x_data = {}

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
plt.ion()
plt.show()
for i in range(1000):
	batch_xs, batch_ys = input_pipeline(filenames, batch_size, read_threads, num_epochs=None)
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.8})
	if i % 50 == 0:
		val = compute_accuracy(mnist.test.images, mnist.test.labels)
		print(val)
		y_data[i/50] = val
		x_data[i/50] = i/50
		lines = ax.plot(i/50, val, 'r-', lw=5)

for i in range(20):
	scat = plt.scatter(x_data[i], y_data[i])
	
plt.pause(10000)



