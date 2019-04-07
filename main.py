#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

from tqdm import tqdm

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, l3, l4, l7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # We add a 1x1 convolutions on top of the VGG to reduce the number of filters
    # from 4096 to the number of classes for our specific model.
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                       num_classes,
                                       1, # kernel size
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out,
                                       num_classes,
                                       1, # kernel size
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out,
                                       num_classes,
                                       1, # kernel size
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Up-sampling
    layer7_conv_1x1_up = tf.layers.conv2d_transpose(layer7_conv_1x1,
                                                    num_classes,
                                                    4, # kernel size
                                                    2, # stride
                                                    padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # skip connections
    # we combine the output of the previous layer with the result of the pooling layer 4
    # through element wise addition
    layer4_skip = tf.add(layer7_conv_1x1_up, layer4_conv_1x1)

    # We can then follow this with another transposed convolution layer.
    # Up-sample it by 2
    layer4_skip_up = tf.layers.conv2d_transpose(layer4_skip,
                                                num_classes,
                                                4, # kernel size
                                                strides=(2, 2),
                                                padding='same',
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # we combine the output of the previous layer with the result of the pooling layer 3
    # through element wise addition
    layer3_skip = tf.add(layer4_skip_up, layer3_conv_1x1)

    # We can then follow this with another transposed convolution layer.
    # Up-sample it by 8
    output = tf.layers.conv2d_transpose(layer3_skip,
                                        num_classes,
                                        16, # kernel size
                                        strides=(8, 8),
                                        padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # the output tensor is 4D so we have to reshape it to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # logits is now a 2D tensor where each row represents a pixel and each column a class.
    # From here we can just use standard cross entropy loss:
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)

    # Returns the index (the class) with the largest value
    predicted_label = tf.argmax(logits, axis=-1)
    sparse_correct_label = tf.argmax(correct_label, axis=-1)

    with tf.variable_scope("iou") as scope:
        iou, iou_op = tf.metrics.mean_iou(
            sparse_correct_label, predicted_label, num_classes)

    metric_vars = [v for v in tf.local_variables()
                   if v.name.split('/')[0] == 'iou']

    metric_reset_ops = tf.variables_initializer(metric_vars)

    return logits, training_operation, cross_entropy_loss, iou, iou_op, metric_reset_ops
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou_op, iou, metric_reset_ops):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    """
    Tensor Board Variables
    """
    train_loss_summary = tf.placeholder(tf.float32)
    train_iou_summary = tf.placeholder(tf.float32)

    tf.summary.scalar("train_loss", train_loss_summary)
    tf.summary.scalar("train_iou", train_iou_summary)

    # val_loss_summary = tf.placeholder(tf.float32)
    # val_iou_summary = tf.placeholder(tf.float32)
    # tf.summary.scalar("val_loss", val_loss_summary)
    # tf.summary.scalar("val_iou", val_iou_summary)

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('charts/kitti_epoch_100_batch_28_keep_0_5_larn_rate_0_0009_augmented', graph=sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    learning_rate_val = 0.0009
    keep_prob_val = 0.5

    is_variable_available = len(tf.trainable_variables()) > 0

    if is_variable_available:
        saver = tf.train.Saver()

    epoch_progress_bar = tqdm(range(epochs))

    for epoch in epoch_progress_bar:
        train_loss_sum = 0.0
        batch_count = 0

        print("EPOCH {} ...".format(epoch  + 1))
        for image, label in get_batches_fn(batch_size):
            _, loss, _ = sess.run([train_op, cross_entropy_loss, iou_op],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: keep_prob_val, learning_rate: learning_rate_val})
            train_loss_sum += loss
            batch_count += 1

        if is_variable_available:
            saver.save(sess, "./model/SemanticSegmentationTfModel")

        # Evaluate Progress
        train_iou = sess.run(iou)
        train_loss = train_loss_sum / batch_count

        sess.run(metric_reset_ops)

        epoch_progress_bar.write(
            "Epoch %03d: train loss: %.4f train IoU: %.4f"
            % (epoch, train_loss, train_iou))

        summary_val = sess.run(
            summary, feed_dict={train_loss_summary: train_loss,
                                train_iou_summary: train_iou})


        '''
        val_loss_sum = 0.0
        batch_count = 0

        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([iou_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 1.0})
            val_loss_sum += loss
            batch_count += 1

        val_iou = sess.run(iou)
        val_loss = val_loss_sum / batch_count

        epoch_progress_bar.write(
            "Epoch %03d: loss: %.4f mIoU: %.4f val_loss: %.4f val_mIoU: %.4f"
            % (epoch, train_loss, train_iou, val_loss, val_iou))
            
            

        summary_val = sess.run(
            summary, feed_dict={train_loss_summary: train_loss,
                                val_loss_summary: val_loss,
                                train_iou_summary: train_iou,
                                val_iou_summary: val_iou})
        
        '''

        writer.add_summary(summary_val, epoch)


tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    epochs = 100
    batch_size = 4 # 4 * 7

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss, iou, iou_op, metric_reset_ops = optimize(
            nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, iou_op, iou, metric_reset_ops)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
