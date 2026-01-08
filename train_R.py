# coding=utf-8
"""
Rotation-only training entry with selectable encoder (siamese | cvt).
This wraps the DirectionNet-R training for 9D/6D with identical losses/logging.
Also includes optional parameter/FLOPs profiling on a dummy batch.
"""
import time
from absl import app, flags
import tensorflow.compat.v1 as tf
import dataset_loader
import model
import losses
import util

flags.DEFINE_string('data_dir', '', 'Training data directory')
flags.DEFINE_string('checkpoint_dir', '', 'Where to save checkpoints/summaries')
flags.DEFINE_integer('batch', 4, 'Batch size')
flags.DEFINE_integer('n_epoch', 2, 'Epochs to repeat the dataset')
flags.DEFINE_integer('max_steps', None, 'Maximum training steps (overrides n_epoch if set)')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_string('rep', '9D', 'Representation: 9D | 6D')
flags.DEFINE_string('encoder', 'siamese', 'Encoder: siamese | cvt')
flags.DEFINE_integer('distribution_height', 64, 'Output distribution height')
flags.DEFINE_integer('distribution_width', 64, 'Output distribution width')
flags.DEFINE_float('alpha', 8e7, 'Weight for distribution loss')
flags.DEFINE_float('beta', 0.1, 'Weight for spread loss')
flags.DEFINE_float('kappa', 10., 'vMF concentration for targets')
flags.DEFINE_bool('profile_model', False, 'Print params/FLOPs and exit')

FLAGS = flags.FLAGS

tf.disable_eager_execution()


def build_graph(src_img, trt_img, rotation_gt, n_out, encoder):
  net = model.DirectionNet(n_out, encoder_type=encoder)
  pred = net(src_img, trt_img, training=True)
  directions, expectation, distribution_pred = util.distributions_to_directions(pred)
  directions_gt = rotation_gt[:, :n_out]
  if n_out == 3:
    rotation_estimated = util.svd_orthogonalize(directions)
  else:
    rotation_estimated = util.gram_schmidt(directions)

  distribution_gt = util.spherical_normalization(util.von_mises_fisher(
      directions_gt, tf.constant(FLAGS.kappa, tf.float32),
      [FLAGS.distribution_height, FLAGS.distribution_width]), rectify=False)

  direction_loss = losses.direction_loss(directions, directions_gt)
  distribution_loss = tf.constant(FLAGS.alpha, tf.float32) * losses.distribution_loss(distribution_pred, distribution_gt)
  spread_loss = tf.cast(FLAGS.beta, tf.float32) * losses.spread_loss(expectation)
  loss = direction_loss + distribution_loss + spread_loss

  rotation_error = tf.reduce_mean(util.rotation_geodesic(rotation_estimated, rotation_gt))

  tf.summary.scalar('loss', loss)
  tf.summary.scalar('rotation_error', util.radians_to_degrees(rotation_error))

  opt = tf.train.GradientDescentOptimizer(FLAGS.lr)
  global_step = tf.train.get_or_create_global_step()
  train_op = opt.minimize(loss, global_step=global_step)
  update_op = net.updates
  return tf.group([train_op, update_op]), loss, global_step


def profile_once():
  with tf.Graph().as_default():
    # Dummy inputs
    src = tf.zeros([2, 256, 256, 3], tf.float32)
    trt = tf.zeros([2, 256, 256, 3], tf.float32)
    rot = tf.eye(3, batch_shape=[2])
    n_out = 3 if FLAGS.rep == '9D' else 2
    net = model.DirectionNet(n_out, encoder_type=FLAGS.encoder)
    pred = net(src, trt, training=False)

    # Params
    total_params = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # FLOPs (best-effort; may undercount custom ops)
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(tf.get_default_graph(), options=opts)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      p_count = sess.run(total_params)
      print('Model parameters: {:,}'.format(int(p_count)))
      if flops is not None:
        print('Approx FLOPs: {:,}'.format(int(flops.total_float_ops)))
      else:
        print('FLOPs not available')


def main(_):
  if FLAGS.profile_model:
    profile_once()
    return

  n_out = 3 if FLAGS.rep == '9D' else 2
  ds = dataset_loader.data_loader(data_path=FLAGS.data_dir,
                                  epochs=FLAGS.n_epoch,
                                  batch_size=FLAGS.batch,
                                  training=True,
                                  load_estimated_rot=False)
  elems = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
  src_img, trt_img = elems.src_image, elems.trt_image
  rotation_gt = elems.rotation

  train_op, loss, step = build_graph(src_img, trt_img, rotation_gt, n_out, FLAGS.encoder)

  merged_summary = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, tf.get_default_graph())

  class TimingHook(tf.train.SessionRunHook):
    def begin(self):
      self.timing_log = []
    def before_run(self, run_context):
      self.start = time.time()
    def after_run(self, run_context, run_values):
      self.timing_log.append(time.time() - self.start)

  hooks = [TimingHook(), tf.train.StepCounterHook(), tf.train.NanTensorHook(loss)]
  
  # Add StopAtStepHook if max_steps is specified
  if FLAGS.max_steps is not None:
    hooks.append(tf.train.StopAtStepHook(last_step=FLAGS.max_steps))

  with tf.train.MonitoredTrainingSession(
      is_chief=True,
      checkpoint_dir=FLAGS.checkpoint_dir,
      hooks=hooks,
      save_checkpoint_steps=2000,
      save_summaries_secs=180) as sess:
    tf.logging.info('Training session started, beginning training loop...')
    while not sess.should_stop():
      tf.logging.info('About to run training step...')
      _, l, s, summary = sess.run([train_op, loss, step, merged_summary])
      summary_writer.add_summary(summary, s)
      if s % 10 == 0:
        tf.logging.info('step = %d, loss = %.5f', s, l)

if __name__ == '__main__':
  app.run(main)
