import sys
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from models.config import cfg
from data_input.utils import load_data


def train(model, supervisor):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    train_batch = tf.data.Dataset.from_tensor_slices((trX, trY)).batch(cfg.batch_size).repeat().make_one_shot_iterator().get_next()
    valid_batch = tf.data.Dataset.from_tensor_slices((valX, valY)).batch(cfg.batch_size).repeat().make_one_shot_iterator().get_next()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.logdir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(cfg.logdir + '/valid')

        # Print stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        for epoch in range(cfg.epoch):
            sys.stdout.write('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            sys.stdout.flush()
            if supervisor.should_stop():
                print('supervisor stopped!')
                break

            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                global_step = epoch * num_tr_batch + step
                tr_x_batch, tr_y_batch = sess.run(train_batch)
                val_x_batch, val_y_batch = sess.run(valid_batch)

                if global_step % cfg.train_sum_freq == 0:
                    # train
                    _, loss, train_acc, summary_str = sess.run(
                        [model.train_op, model.loss, model.accuracy, model.train_summary],
                        {model.images: tr_x_batch, model.labels: tr_y_batch})
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    train_writer.add_summary(summary_str, global_step)
                    # valid
                    valid_acc, summary_str = sess.run([model.accuracy, model.train_summary],
                                                      {model.images: val_x_batch, model.labels: val_y_batch})
                else:
                    sess.run(model.train_op, {model.images: tr_x_batch, model.labels: tr_y_batch})

            val_acc = 0
            for i in range(num_val_batch):
                start = i * cfg.batch_size
                end = start + cfg.batch_size
                acc, summary_str = sess.run([model.accuracy, model.train_summary],
                                            {model.X: valX[start:end], model.labels: valY[start:end]})
                val_acc += acc
            val_acc = val_acc / (cfg.batch_size * num_val_batch)
            print('acc on all valid set: ', val_acc)

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        train_writer.close()
        valid_writer.close()


def evaluation(model, supervisor):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
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

    if cfg.dataset == 'mnist' or cfg.dataset == 'fashion-mnist':
        tf.logging.info(' Loading Graph...')
        model = Model(height=28, width=28, channels=1, num_label=10)
    elif cfg.dataset == 'smallNORB':
        model = Model(height=32, width=32, channels=3, num_label=5)
    elif cfg.dataset == 'cifar10':
        model = Model(height=32, width=32, channels=3, num_label=10)
    else:
        raise EnvironmentError('no such data set')

    tf.logging.info(' Graph loaded')

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start trainging...')
        train(model, sv)
        tf.logging.info('Training done')
    else:
        evaluation(model, sv)


if __name__ == "__main__":
    tf.app.run()
