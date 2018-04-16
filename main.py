import sys

import os
import tensorflow as tf
from tqdm import tqdm

from data_input.utils import create_train_set, create_test_set
from config import cfg


def train(model):
    tf.reset_default_graph()
    with model.graph.as_default():
        handle = tf.placeholder(tf.string, [])
        images, labels, train_iterator, val_iterator, num_label, num_batch = create_train_set(cfg.dataset, handle,
                                                                                              cfg.batch_size)

        model((images, labels), num_label=num_label, is_training=True, distort=cfg.distort,
              standardization=cfg.standardization)

        tf.logging.info(' Graph loaded')

        ckpt = tf.train.get_checkpoint_state(cfg.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            global_step = int(ckpt_name.split('-')[-1])
            last_epoch = global_step // num_batch
            last_step = global_step % num_batch
        else:
            global_step = 0
            last_epoch = 0
            last_step = 0

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)
            train_writer = tf.summary.FileWriter(cfg.logdir + '/train', sess.graph)
            valid_writer = tf.summary.FileWriter(cfg.logdir + '/valid')

            train_handle = sess.run(train_iterator.string_handle())
            val_handle = sess.run(val_iterator.string_handle())

            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Print stats
            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                model.graph,
                tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

            for epoch in range(last_epoch, cfg.epoch):
                sys.stdout.write('Training for epoch ' + str(epoch+1) + '/' + str(cfg.epoch) + ':')
                sys.stdout.flush()

                bar = tqdm(range(last_step, num_batch), total=num_batch, ncols=100, leave=False, unit='b')
                for _ in bar:
                    if global_step % cfg.save_summaries_steps == 0:
                        # train
                        _, train_acc, summary_str = sess.run(
                            [model.train_op, model.accuracy, model.merged_summary],
                            feed_dict={handle: train_handle})
                        train_writer.add_summary(summary_str, global_step)
                        # valid
                        sess.run(val_iterator.initializer)
                        valid_acc, summary_str = sess.run([model.accuracy, model.merged_summary],
                                                          feed_dict={handle: val_handle})
                        valid_writer.add_summary(summary_str, global_step)
                        bar.set_description('tr_acc:{} val_acc:{}'.format(train_acc, valid_acc))
                    else:
                        sess.run(model.train_op, feed_dict={handle: train_handle})

                    global_step += 1
                    if global_step % cfg.save_checkpoint_steps == 0:
                        saver.save(sess, cfg.logdir + '/model.ckpt', global_step=global_step)

            train_writer.close()
            valid_writer.close()


def evaluation(model):
    with model.graph.as_default():
        images, labels, num_label = create_test_set(cfg.dataset, cfg.batch_size)
        model((images, labels), num_label=num_label, is_training=False)
        saver = tf.train.Saver()
    num_te_batch = len(labels) / cfg.batch_size

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for _ in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            acc = sess.run(model.accuracy)
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        print('acc on all test set: ', test_acc)


def main(_):
    if cfg.model == 'vector':
        from models.vector_caps_model import CapsNet as Model
    elif cfg.model == 'matrix':
        from models.matrix_caps_model import CapsNet as Model
    else:
        from models.baseline import Model

    model = Model()

    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model)
        tf.logging.info('Training done')
    else:
        evaluation(model)


if __name__ == "__main__":
    tf.app.run()
