import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display, Image, HTML
import cv2
import tensorflow as tf

#训练数据与测试数据所在文件夹
TRAIN_DIR = 'C:\\Users\\longyiyuan\\Desktop\\train_data\\'
TEST_DIR = 'C:\\Users\\longyiyuan\\Desktop\\test_data\\'

#最终放到模型中训练的图像的大小为64*64
IMAGE_SIZE = 96
#图像的厚度，rgb为3层，灰度图像为1层
CHANNELS = 3
#像素的深度，最大为255，在预处理中会用到
pixel_depth = 255.0

#输出文件目录，保存当前的配置
OUTFILE = '../smalltest.npsave.bin'
#训练文件中狗的数量
TRAINING_AND_VALIDATION_SIZE_DOGS = 10000
#训练文件中猫的数量
TRAINING_AND_VALIDATION_SIZE_CATS = 10000
#训练文件中所有图片的数量
TRAINING_AND_VALIDATION_SIZE_ALL = 20000
#测试文件中狗的数量
TEST_SIZE_DOGS = 250
#测试文件中猫的数量
TEST_SIZE_CATS = 250
#训练数据的大小，也就是所有图片的数量
TRAINING_SIZE = 20000
#测试数据的大小，也就是测试所有图片的数量
TEST_SIZE_ALL = 500

#从训练文件夹中获得训练图片的路径列表
train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]
#从训练文件夹中获得训练数据的狗的图片的路径列表
train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
#从训练文件夹中获得训练数据中猫的图片的路径列表
train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

#从测试文件夹中获的测试图片的路径列表
test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
#从测试文件中获得训练数据的狗的图片的路径列表
test_dogs = [TEST_DIR + i for i in os.listdir(TEST_DIR) if 'dog' in i]
#从测试文件中获的训练数据的猫的图片的路径列表
test_cats = [TEST_DIR + i for i in os.listdir(TEST_DIR) if 'cat' in i]

#将训练数据中狗和猫的图片数组拼接起来
train_images = train_dogs[:TRAINING_AND_VALIDATION_SIZE_DOGS] + train_cats[:TRAINING_AND_VALIDATION_SIZE_CATS]
#将构建训练数据标签数组，因为前面的全是狗，后面的全是猫
train_labels = np.array((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + ['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS)

#将测试数据中狗和猫的图片数组拼接起来
test_images = test_dogs[:TEST_SIZE_DOGS] + test_cats[:TEST_SIZE_CATS]
#构建测试数据标签数组，同样是狗在前，猫在后面
test_labels = np.array((['dogs'] * TEST_SIZE_DOGS + ['cats'] * TEST_SIZE_CATS))

#更改图片的大小变成规定的大小函数
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    img2 = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)

    return img3[:,:,::-1]

#正则化图片数据函数
def prep_data(images):
	count = len(images)
	data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)

	for i, image_file in enumerate(images):
		#调用上面的read_image函数对图片的大小进行修改
		img = read_image(image_file)
		#将图片数据转换成浮点型数组
		image_data = np.array(img, dtype=np.float32)
		#讲图片的每层都进行正则化
		image_data[:, :, 0] = (image_data[:, :, 0].astype(float) - pixel_depth / 2) / pixel_depth
		image_data[:, :, 1] = (image_data[:, :, 1].astype(float) - pixel_depth / 2) / pixel_depth
		image_data[:, :, 2] = (image_data[:, :, 2].astype(float) - pixel_depth / 2) / pixel_depth

		data[i] = image_data
		if i % 1000 == 0:
			print("Processed {} of {}".format(i, count))

	return data

#正则化训练数据和测试数据
train_normalized = prep_data(train_images)
test_normalized = prep_data(test_images)

#输出训练数据和测试数据的维度
print("Train shape: {}".format(train_normalized.shape))
print("Test shape: {}".format(test_normalized.shape))

#初始化随机种子
np.random.seed(133)
#随机函数，对标签和图片按照同样的随机顺序，进行随机
def randomize(dataset, labels):
	#获的随机顺序
	permutation = np.random.permutation(labels.shape[0])
	#将数据按照顺序随机
	shuffled_dataset = dataset[permutation, :, :, :]
	#将标签按照顺序随机
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels
#将训练数据和训练标签随机
train_dataset_rand, train_labels_rand = randomize(train_normalized, train_labels)
#将测试数据和标签随机
test_dataset, test_labels = randomize(test_normalized, test_labels)

#获得随机化后的训练数据
train_dataset = train_dataset_rand[:TRAINING_SIZE, :, :, :]
train_labels = train_labels_rand[:TRAINING_SIZE]
#获的随机化后的测试数据
test_dataset = train_dataset_rand[:TEST_SIZE_ALL, :, :, :]
test_labels = train_labels_rand[:TEST_SIZE_ALL]

#输出训练和测试数据的维度
print('Training', train_dataset.shape, train_labels.shape)
print('Test', test_dataset.shape, test_labels.shape)

#显示训练数据正则化后的第一个图片
plt.imshow (train_normalized[0,:,:,:], interpolation='nearest')
plt.figure ()
#显示测试数据正则化的第一个图片
plt.imshow (test_normalized[0, :, :, :], interpolation='nearest')
plt.figure()


image_size = IMAGE_SIZE
num_labels = 2
num_channels = 3 # rgb

#将训练数据转化成为模型需要的数据类型
def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  #狗是0，猫是1
  labels = (labels=='cats').astype(np.float32);
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

#调动上述函数，对数据转换成模型要求的格式，即标签为[0,1]类型，
#数据类型为[-1,imagesize, imageszie, channels]
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print ('Training set', train_dataset.shape, train_labels.shape)
print ('Test set', test_dataset.shape, test_labels.shape)

#定义变量
xs = tf.placeholder(tf.float32, [None, 96, 96, 3])
ys = tf.placeholder(tf.float32, [None, 2])
#定义保存的概率
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 96, 96, 3])

#权值初始化计算函数
def Weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


#偏置初始化函数
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#卷积计算函数
def  con2d(inputs, weight):
	return tf.nn.conv2d(inputs, weight, strides=[1,1,1,1], padding='SAME')

#池化层计算函数
def max_pooling_2x2(inputs):
	return tf.nn.max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#准确度计算函数
def computer_accuracy(v_xs, v_ys):
	global prediction
	#计算得到预测值
	y_re = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	#如果预测值中1的位置和y中1的位置相同就是正确的
	return (np.sum(np.argmax(y_re, 1) == np.argmax(v_ys, 1)) / 500)


#第一层卷积层
W_conv1 = Weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(con2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pooling_2x2(h_conv1)

#第二层卷积层
W_conv2 = Weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)

#第三层卷基层
W_conv3 = Weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(con2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pooling_2x2(h_conv3)

print("The shape of the h_pool3 is {}".format(h_pool3.shape))
print("The shape of the h_conv3 is {}".format(h_conv3.shape))
#全连接层
w_f1 = Weight_variable([12*12*128, 2048])
b_f1 = bias_variable([2048])
h_pool3_flat = tf.reshape(h_pool3, [-1, 12*12*128])
h_f1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_f1) + b_f1)
h_f1_drop = tf.nn.dropout(h_f1, keep_prob)

w_f2 = Weight_variable([2048, 2])
b_f2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_f1_drop, w_f2) + b_f2)

#误差计算
cross_entropy = tf.reduce_mean(-tf.reduce_sum(-ys*tf.log(prediction),
	reduction_indices=[1]))

#采用梯度下降法进行训练
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

#初始化变量
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(2000):
	#每次训练500个，计算的是每次训练完的最后标志位
	offset = (i * 500) % (train_labels.shape[0] - 500)
	batch_xs = train_dataset[offset:(offset + 500), :, :, :]
	batch_ys = train_labels[offset:(offset + 500), :]
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.8})
    #每五十步计算一次准确率
	if i % 50 == 0:
		val = computer_accuracy(test_dataset, test_labels)
		print("Train step %d:" %i ,val)

plt.pause(10000)
