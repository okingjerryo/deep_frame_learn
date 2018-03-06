import tensorflow as tf
import tensorrt as trt
import numpy as np

# 可以将 maxrange 的范围自定义
max_range = tf.placeholder(tf.int64,shape=[],name='max_range')
data_handle = tf.placeholder(tf.string,shape=[],name='data_handle')

def trainfun(x):
    return x+tf.random_uniform([],-10,10,tf.int64)
def reinitializable_itor_lr():

    # 很多时候，dataset 可以使用reuseItor ，方便 dataset 获取验证集和训练集。
    # 给 dataset 加入 map 属性可以对生成数据进行预处理函数定义
    training_dataset = tf.data.Dataset.range(max_range).map(
        lambda x: trainfun(x))
    validation_dataset = tf.data.Dataset.range(max_range)
    # 通过数据 train&vail 集合 shape 和 type 定义 iter
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    # 定义取得下一元素方法
    next_element = iterator.get_next()

    #最终定义训练集和验证集操作
    tranining_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    return (tranining_init_op,validation_init_op,next_element)
# test 
def read_data_reinit():
    tr_op,vl_op,next_el = reinitializable_itor_lr()
    #开始读利用 dataset 取数据
    with tf.Session() as sess:
        for _ in range(20):
            sess.run(tr_op,feed_dict={max_range:100})
            for _ in range(100):
                sess.run(next_el)

            sess.run(vl_op,feed_dict={max_range:70})
            for _ in range(50):
                sess.run(next_el)

def handle_itor_lr():
    # 创建train 数据和验证数据
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x:trainfun(x)
    )
    validate_dataset = tf.data.Dataset.range(50)


    # 创建handle_itor
    iterator = tf.data.Iterator.from_string_handle(data_handle,training_dataset.output_types,training_dataset.output_shapes)
    next_elem = iterator.get_next()

    #分离不同的itor 计算节点给外部程序使用

def read_data_handle():
    pass

if __name__ == '__main__':
    read_data_handle()
