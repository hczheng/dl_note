# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:46:59 2016

@author: spark
"""

import tensorflow as tf
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))
#a = tf.constant(10)
#b = tf.constant(32)
#print(sess.run(a + b))

#矩阵计算
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])
# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

#创建一个矩阵乘法 matmul op,把matrix1, matrix2作为输入
#product代表结果
product = tf.matmul(matrix1, matrix2)

# 启动默认图
sess = tf.Session()
#调用sess的run方法
#result = sess.run(product)
#print(result)
#sess.close()

#session对象在用完后需要关闭以释放资源，除了显示调用close外，还可以使用with代码块：
with tf.Session() as sess:
  result = sess.run([product])
  print(result)

#f you have more than one GPU available on your machine, to use a GPU beyond the 
#first you must assign ops to it explicitly. 
#Use with...Device statements to specify which CPU or GPU to use for operations:
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)

#交互式使用
# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2. -1.]

# Close the Session when we're done.
sess.close()

#变量，注意，只有在run的时候才真正赋值，在run之前assign并没有赋值操作
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
  # Run the 'init' op
  sess.run(init_op)
  # Print the initial value of 'state'
  print(sess.run(state))
  # Run the op that updates 'state' and print 'state'.
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))


#Fetches：取回多个tensor，需要获取的多个tensor值，在op的一次运行中一起获得（而不是逐个去获取）
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)

# output:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

#Feeds
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]

