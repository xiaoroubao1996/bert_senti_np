import tensorflow as tf
from bert import modeling
import os
import args


def bert_model():
    bert_config = modeling.BertConfig.from_json_file(args.config_name)# 配置文件地址。
    configsession = tf.ConfigProto()
    configsession.gpu_options.allow_growth = True
    sess = tf.Session(config=configsession)
    input_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_mask")
    segment_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="segment_ids")

    with sess.as_default():
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())# 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
        saver.restore(sess, args.ckpt_name)
    return sess, model