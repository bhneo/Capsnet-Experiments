import sys
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from models.config import cfg
from data_input.utils import create_train_set, create_test_set


def train(model):
    with model.graph.as_default():
        images, labels, train_iterator, val_iterator, handle, num_label, num_batch = create_train_set(cfg.dataset,
                                                                                                      cfg.batch_size)

        model((images, labels), num_label=num_label, is_training=True, distort=cfg.distort,
              standardization=cfg.standardization)
    tf.logging.info(' Graph loaded')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    with sv.managed_session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.logdir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(cfg.logdir + '/valid')

        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        # Print stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        for epoch in range(1, cfg.epoch+1):
            sys.stdout.write('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            sys.stdout.flush()
            global_step = 0
            if sv.should_stop():
                print('supervisor stopped!')
                break

            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                global_step = epoch * num_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    # train
                    _, loss, train_acc, summary_str = sess.run(
                        [model.train_op, model.loss, model.accuracy, model.train_summary],
                        {handle: train_handle})
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    train_writer.add_summary(summary_str, global_step)
                    # valid
                    valid_acc, summary_str = sess.run([model.accuracy, model.merged_summary],
                                                      {handle: val_handle})
                    valid_writer.add_summary(summary_str, global_step)
                else:
                    sess.run(model.train_op, {handle: train_handle})

            if (epoch + 1) % cfg.save_freq == 0:
                sv.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        train_writer.close()
        valid_writer.close()


def evaluation(model):
    with model.graph.as_default():
        images, labels, num_label = create_test_set(cfg.dataset, cfg.batch_size)
        model((images, labels), num_label=num_label, is_training=False)
    num_te_batch = len(labels) / cfg.batch_size
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
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
