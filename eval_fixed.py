#!/usr/bin/env python3
# coding=utf-8
"""
Fixed evaluation script for DirectionNet that works properly with the dataset.
"""
from absl import app
from absl import flags
import dataset_loader
import model
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import util
import os

tf.compat.v1.disable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string('eval_data_dir', '', 'The test data directory.')
flags.DEFINE_string('save_summary_dir', '', 'The directory to save summary.')
flags.DEFINE_string('checkpoint_dir', '', 'The directory to load the checkpoints.')
flags.DEFINE_string('model', '9D', '9D(rotation), T(translation), Single(DirectionNet-Single)')
flags.DEFINE_integer('batch', 1, 'Size of mini-batches.')
flags.DEFINE_integer('testset_size', 100, 'The size of the test set.')
flags.DEFINE_integer('distribution_height', 64, 'The height dimension of output distributions.')
flags.DEFINE_integer('distribution_width', 64, 'The width dimension of output distributions.')
flags.DEFINE_integer('transformed_height', 344, 'The height dimension of input images after derotation transformation.')
flags.DEFINE_integer('transformed_width', 344, 'The width dimension of input images after derotation transformation.')
flags.DEFINE_float('kappa', 10., 'A coefficient multiplied by the concentration loss.')
flags.DEFINE_float('transformed_fov', 105., 'The field of view of input images after derotation transformation.')
flags.DEFINE_bool('derotate_both', True, 'Derotate both input images when training DirectionNet-T')
#flags.DEFINE_string('encoder', 'siamese', 'Encoder: siamese | cvt')
flags.DEFINE_string(
    'encoder',
    'coarse',
    'Encoder backbone: siamese | cvt | coarse (coarse_transformer)')

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Create dataset
    print(f"Loading dataset from: {FLAGS.eval_data_dir}")
    ds = dataset_loader.data_loader(
        data_path=FLAGS.eval_data_dir,
        epochs=1,
        batch_size=FLAGS.batch,
        training=False,
        load_estimated_rot=FLAGS.model == 'T'
    )

    # Create iterator
    iterator = tf.compat.v1.data.make_initializable_iterator(ds)
    elements = iterator.get_next()
    src_img, trt_img = elements.src_image, elements.trt_image
    rotation_gt = elements.rotation
    translation_gt = elements.translation
    
    # Normalize encoder selection for DirectionNet
    encoder_aliases = {
        'coarse_transformer': 'coarse_transformer',
        'coarse': 'coarse_transformer',
        'cvt': 'cvt',
        'cross_view_transformer': 'cvt',
        'siamese': 'siamese'
    }

    encoder_key = FLAGS.encoder.lower()
    if encoder_key not in encoder_aliases:
        raise app.UsageError(
            f"Unknown encoder '{FLAGS.encoder}'. Use siamese, cvt, or coarse.")

    encoder_type = encoder_aliases[encoder_key]
    # Build model based on type
    print(f"Building model: {FLAGS.model}")
    if FLAGS.model == '9D' or FLAGS.model == '6D':
        n_distributions = 3 if FLAGS.model == '9D' else 2
        #net = model.DirectionNet(n_distributions, encoder_type=FLAGS.encoder)
        net = model.DirectionNet(n_distributions, encoder_type=encoder_type)
        pred = net(src_img, trt_img, training=False)
        directions, _, _ = util.distributions_to_directions(pred)
        
        if n_distributions == 3:
            rotation_estimated = util.svd_orthogonalize(directions)
        else:
            rotation_estimated = util.gram_schmidt(directions)
        
        # Calculate errors
        angular_errors = util.angular_distance(directions, rotation_gt[:, :n_distributions])
        x_error = tf.reduce_mean(angular_errors[:, 0])
        y_error = tf.reduce_mean(angular_errors[:, 1])
        z_error = tf.reduce_mean(angular_errors[:, 2])
        rotation_error = tf.reduce_mean(util.rotation_geodesic(rotation_estimated, rotation_gt))
        
        # Metrics to track
        metrics = {
            'angular_error_x': x_error,
            'angular_error_y': y_error,
            'angular_error_z': z_error,
            'rotation_error': rotation_error
        }
        
    elif FLAGS.model == 'T':
        # Translation model
        fov_gt = tf.squeeze(elements.fov, -1)
        rotation_pred = elements.rotation_pred
        
        #net = model.DirectionNet(1, encoder_type=FLAGS.encoder)
        net = model.DirectionNet(1, encoder_type=encoder_type)
        (transformed_src, transformed_trt) = util.derotation(
            src_img, trt_img, rotation_pred, fov_gt,
            FLAGS.transformed_fov,
            [FLAGS.transformed_height, FLAGS.transformed_width],
            FLAGS.derotate_both
        )
        
        pred = net(transformed_src, transformed_trt, training=False)
        directions, _, _ = util.distributions_to_directions(pred)
        
        translation_gt_expanded = tf.expand_dims(translation_gt, 1)
        half_derotation = util.half_rotation(rotation_pred)
        directions = tf.matmul(directions, half_derotation, transpose_b=True)
        translation_error = tf.reduce_mean(tf.acos(tf.clip_by_value(
            tf.reduce_sum(directions * translation_gt_expanded, -1), -1., 1.)))
        
        metrics = {'translation_error': translation_error}
        
    elif FLAGS.model == 'Single':
        # Single model (both rotation and translation)
        #net = model.DirectionNet(4, encoder_type=FLAGS.encoder)
        net = model.DirectionNet(4, encoder_type=encoder_type)
        pred = net(src_img, trt_img, training=False)
        directions, _, _ = util.distributions_to_directions(pred)
        rotation_estimated = util.svd_orthogonalize(directions[:, :3])
        
        translation_gt_expanded = tf.expand_dims(translation_gt, 1)
        directions_gt = tf.concat([rotation_gt, translation_gt_expanded], 1)
        
        angular_errors = util.angular_distance(directions, directions_gt)
        x_error = tf.reduce_mean(angular_errors[:, 0])
        y_error = tf.reduce_mean(angular_errors[:, 1])
        z_error = tf.reduce_mean(angular_errors[:, 2])
        translation_error = tf.reduce_mean(angular_errors[:, 3])
        rotation_error = tf.reduce_mean(util.rotation_geodesic(rotation_estimated, rotation_gt))
        
        metrics = {
            'angular_error_x': x_error,
            'angular_error_y': y_error,
            'angular_error_z': z_error,
            'rotation_error': rotation_error,
            'translation_error': translation_error
        }
    
    # Create session
    print("Creating session...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Saver for loading checkpoint
    saver = tf.train.Saver()
    
    # Summary writer
    if FLAGS.save_summary_dir:
        os.makedirs(FLAGS.save_summary_dir, exist_ok=True)
        summary_writer = tf.summary.FileWriter(FLAGS.save_summary_dir)
    
    with tf.Session(config=config) as sess:
        # Load checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_checkpoint:
            print(f"Restoring checkpoint: {latest_checkpoint}")
            saver.restore(sess, latest_checkpoint)
        else:
            print("ERROR: No checkpoint found!")
            return
        
        # Initialize iterator
        sess.run(iterator.initializer)
        
        # Run evaluation
        print(f"Evaluating {FLAGS.testset_size} samples...")
        num_steps = FLAGS.testset_size // FLAGS.batch
        
        # Accumulate metrics
        accumulated_metrics = {key: [] for key in metrics.keys()}
        
        try:
            for step in range(num_steps):
                metric_values = sess.run(metrics)
                
                for key, value in metric_values.items():
                    accumulated_metrics[key].append(value)
                    
                # # Write per-step summaries so TensorBoard shows a curve instead of a single point.
                # if FLAGS.save_summary_dir:
                #     summary = tf.Summary()
                #     for key, value in metric_values.items():
                #         if 'error' in key:
                #             value_deg = util.radians_to_degrees(value)
                #             summary.value.add(tag=f'eval/{key}_degrees', simple_value=value_deg)
                #             summary.value.add(tag=f'eval/{key}_radians', simple_value=value)
                #         else:
                #             summary.value.add(tag=f'eval/{key}', simple_value=value)
                #     summary_writer.add_summary(summary, step)

                if (step + 1) % 10 == 0:
                    print(f"Step {step + 1}/{num_steps}")
                    
        except tf.errors.OutOfRangeError:
            print(f"Dataset exhausted at step {step}")
        
        # Calculate and print final metrics
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Write summaries to TensorBoard
        if FLAGS.save_summary_dir:
            for key, values in accumulated_metrics.items():
                if len(values) > 0:
                    mean_val = sum(values) / len(values)
                    # Create a summary for this metric
                    summary = tf.Summary()
                    if 'error' in key:
                        mean_val_deg = util.radians_to_degrees(mean_val)
                        summary.value.add(tag=f'eval/{key}_degrees', simple_value=mean_val_deg)
                        summary.value.add(tag=f'eval/{key}_radians', simple_value=mean_val)
                        print(f"{key}: {mean_val_deg:.4f} degrees")
                    else:
                        summary.value.add(tag=f'eval/{key}', simple_value=mean_val)
                        print(f"{key}: {mean_val:.4f}")
                    # Write summary (use step 0 for evaluation metrics)
                    summary_writer.add_summary(summary, 0)
                    # # Write summary using the final evaluation step instead of 0
                    # summary_writer.add_summary(summary, num_steps)
            summary_writer.flush()
            print(f"\n✓ Summaries written to: {FLAGS.save_summary_dir}")
        else:
            for key, values in accumulated_metrics.items():
                if len(values) > 0:
                    mean_val = sum(values) / len(values)
                    if 'error' in key:
                        mean_val_deg = util.radians_to_degrees(mean_val)
                        print(f"{key}: {mean_val_deg:.4f} degrees")
                    else:
                        print(f"{key}: {mean_val:.4f}")
        
        print(f"\nTotal samples evaluated: {sum(len(v) for v in accumulated_metrics.values()) // len(accumulated_metrics)}")
        print("="*60)


if __name__ == '__main__':
    app.run(main)
