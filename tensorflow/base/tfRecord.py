from glob import glob

import tensorflow as tf

# 给read一个类型
TYPE_IMG = 1
TYPE_STR = 2

# 存储record的位置 一定要有最后的斜线
PATH2RECORD = '../resource/tfrecord/'


# 读取图像数据
def read_pic_to_tensor(filename, type):
    # 文件名矩阵
    name_tensor = tf.constant(filename)
    img_str = tf.read_file(name_tensor)
    img_tensor = tf.image.decode_image(img_str)
    if type == TYPE_IMG:
        return img_tensor
    else:
        return img_str


# 定义bytes型数据
def _bytes_feture(bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes]))


# 写为tfrecord的方法
def write_pathimg_to_record(filepath, record_name):
    print('file convert start')
    print('detect', len(filepath), 'files')
    # 定义存储位置和writer
    record_path = PATH2RECORD + record_name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(record_path)
    # 开启sess写入图片
    with tf.Session() as sess:
        for this_path in filepath:
            # 获取当前的图片数据
            imgstr = sess.run(read_pic_to_tensor(this_path, TYPE_STR))
            # 组装为feature_data
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_noise': _bytes_feture(imgstr)
            }))
            writer.write(example.SerializeToString())
    # 关闭writer
    writer.close()
    print('convert end')


if __name__ == '__main__':
    filepath = glob(r'../resource/image/*.*')
    record_name = 'test'
    write_pathimg_to_record(filepath, record_name)
