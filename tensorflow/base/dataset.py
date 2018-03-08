import tensorflow as tf
from tqdm import trange

# 可以将 maxrange 的范围自定义
MAX_RANGE = tf.placeholder(tf.int64, shape=[], name='MAX_RANGE')
DATA_HANDLE = tf.placeholder(tf.string, shape=[], name='DATA_HANDLE')
# record name holder
RECORD_FILE = tf.placeholder(tf.string, shape=[None])
NUMBER_OF_BATCH = 20
def trainfun(x):
    return x+tf.random_uniform([],-10,10,tf.int64)

def reinitializable_itor_lr():

    # 很多时候，dataset 可以使用reuseItor ，方便 dataset 获取验证集和训练集。
    # 给 dataset 加入 map 属性可以对生成数据进行预处理函数定义
    training_dataset = tf.data.Dataset.range(MAX_RANGE).map(
        lambda x: trainfun(x))
    validation_dataset = tf.data.Dataset.range(MAX_RANGE)
    # 通过数据 train&vail 集合 shape 和 type 定义 iter
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    # 定义取得下一元素方法
    next_element = iterator.get_next()

    #最终定义训练集和验证集操作
    tranining_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    return (tranining_init_op,validation_init_op,next_element)

def read_data_reinit():
    tr_op,vl_op,next_el = reinitializable_itor_lr()
    #开始读利用 dataset 取数据
    with tf.Session() as sess:
        for _ in range(20):
            sess.run(tr_op, feed_dict={MAX_RANGE: 100})
            for _ in range(100):
                sess.run(next_el)

            sess.run(vl_op, feed_dict={MAX_RANGE: 70})
            for _ in range(50):
                sess.run(next_el)

def handle_itor_lr():
    # 创建train 数据和验证数据
    # 加入repeat 使得数据可以被循环读取
    training_dataset = tf.data.Dataset.range(500).map(
        lambda x:trainfun(x)
    ).repeat()
    validate_dataset = tf.data.Dataset.range(50).repeat()

    # 创建handle_itor
    iterator = tf.data.Iterator.from_string_handle(DATA_HANDLE, training_dataset.output_types
                                                   , training_dataset.output_shapes)
    next_elem = iterator.get_next()

    #分离不同的itor 计算节点给外部程序使用
    train_itor = training_dataset.make_one_shot_iterator()
    valid_itor = validate_dataset.make_initializable_iterator()

    return train_itor, valid_itor, next_elem


def read_DATA_HANDLE():
    # 读数据时，初始化需要的itor和handle
    train_itor, valid_itor, itor_next = handle_itor_lr()
    with tf.Session() as sess:
        train_handle = sess.run(train_itor.string_handle())
        valid_handle = sess.run(valid_itor.string_handle())
        # initable itor 需要初始化
        # 注意需要在外部初始化itor
        sess.run(valid_itor.initializer)  # 与reinitable 不同，feedable itor 直接在sess中定义即可
        # 模拟 20 epoch
        for i in trange(20):
            for _ in range(200):
                sess.run(itor_next, feed_dict={DATA_HANDLE: train_handle})

            for _ in range(50):
                sess.run(itor_next, feed_dict={DATA_HANDLE: valid_handle})


# 解包feature 并对输出图片做预处理
def process_record(input_example):
    features = {
        'img_noise': tf.FixedLenFeature((), tf.string, default_value='')
    }
    # 使用parse_single_example 解压
    parsed_features = tf.parse_single_example(input_example, features)
    # 这里可以放多次decode，比如cgan任务，需要输入两张图片，可以将两张图片都decode后并在一起返回
    # 类似于 return zip(image,label)
    image = tf.image.decode_image(parsed_features['img_noise'])
    image = tf.random_crop(image, size=[200, 200, 3])  # 做一个基本的random_crop
    return image


# 创建一个dataset handle
def init_record_read_handle():
    # 组成dataset
    record_dataset = tf.data.TFRecordDataset(RECORD_FILE)
    record_dataset = record_dataset.map(
        lambda x: process_record(x)  # 对 imgstr解码
    )
    # 设置dataset的属性
    record_dataset = record_dataset.repeat(NUMBER_OF_BATCH)  # epoch num
    record_dataset = record_dataset.batch(2)  # 定义了batch_size
    # 使用feedable itor,创建string handle
    iterator = tf.data.Iterator.from_string_handle(DATA_HANDLE, record_dataset.output_types
                                                   , record_dataset.output_shapes)
    next_elem = iterator.get_next()
    data_itor = record_dataset.make_initializable_iterator()
    return data_itor, next_elem


# 对tfrecord 进行读取
def record_read_process():
    # 加载数据读取流图
    data_itor, next_elem = init_record_read_handle()
    # 设定记录名
    record_name = ['../resource/tfrecord/test.tfrecord']
    # 监控dataset的信号
    with tf.train.MonitoredTrainingSession() as sess:
        record_handle = sess.run(data_itor.string_handle())
        # 注意别忘init itor,这里 init会将itor置为0 注意在sess运行时init即可
        sess.run(data_itor.initializer, feed_dict={RECORD_FILE: record_name})
        # 启用了MonitroedTrainSession 当dataset发出outofrange时自动停止
        while not sess.should_stop():
            # 此时的next_elem则是一次取一个banch
            get_feature = sess.run(next_elem, feed_dict={DATA_HANDLE: record_handle, RECORD_FILE: record_name})
            print(get_feature.shape)

if __name__ == '__main__':
    record_read_process()
