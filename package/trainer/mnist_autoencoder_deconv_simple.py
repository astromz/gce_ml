'''This script demonstrates how to build a convolutional autoencoder
with Keras and deconvolution layers. It can be run on google's Cloud ML.

This code borrowed and extended the example from: https://blog.keras.io/building-autoencoders-in-keras.html

Ming Zhao, August 10, 2017
'''
import time, argparse
import numpy as np

from keras.layers import Input, Dense, Lambda, Reshape
from keras.layers import Conv2D, Conv2DTranspose, AveragePooling2D, UpSampling2D, BatchNormalization, Activation, MaxPooling2D, Cropping2D
from keras.models import Model

from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.datasets import mnist
from tensorflow.python.lib.io import file_io

import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt


#%%

def conv_block(x, n_channels, kernel_size=3, padding='same', activation='relu',
               dilation_rate=1, batch_norm=True, use_transposed_conv=False):
    '''A 2D convolution block with conv2D (or deconv), relu, batchnorm.
    In principle batch_norm should be applied before non-linear activation. However, it
    has become a trend to have a batch norm layer after an activation layer
    (see: https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md ,
    https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras ,
    https://github.com/fchollet/keras/issues/1802 )
    '''
    global do_batch_norm_before_activation

    if use_transposed_conv is True:
        conv = Conv2DTranspose
    else:
        conv = Conv2D

    if batch_norm is True:
        # use_bias = True if uses batch_norm
        if do_batch_norm_before_activation is True:
            conv_layer = conv(n_channels, kernel_size=kernel_size, use_bias=False,
                                padding=padding, dilation_rate = dilation_rate)(x)
            conv_layer = BatchNormalization()(conv_layer)
            conv_layer  = Activation(activation=activation)(conv_layer)

        else:
            conv_layer = conv(n_channels, kernel_size=kernel_size, activation=activation, use_bias=False,
                                padding=padding, dilation_rate = dilation_rate)(x)
            conv_layer = BatchNormalization()(conv_layer)
    else:
        conv_layer = conv(n_channels, kernel_size=kernel_size, activation=activation, use_bias=True,
                            padding=padding, dilation_rate = dilation_rate)(x)
    return conv_layer



def pool_layer(x, method='max', pool_size=(2,2), padding='same'):
    if method == 'max':
        return MaxPooling2D(pool_size, padding=padding)(x)
    else:
        return AveragePooling2D(pool_size, padding=padding)(x)



def train(job_dir=None, job_id=None,
          use_transposed_conv=True, score_metric='mse', loss='binary_crossentropy',
          learning_rate = 0.001, lr_decay=0.001, optimizer_name='adam', n_epochs=100,
          patience=5, batch_norm_before_activation=True, pool_method='max', **kwargs):
    '''main training function'''

    global do_batch_norm_before_activation
    do_batch_norm_before_activation = batch_norm_before_activation
    print('--> batch_norm_before_activation== {}!!!\n'.format(do_batch_norm_before_activation))

    # input image dimensions
    img_rows, img_cols, img_chns = 28, 28, 1
    original_img_size = (img_rows, img_cols, img_chns)

    n_filters = 8 # number of convolutional filters to use
    kernel_size = 3 # convolution kernel size
    batch_size = 1000

    lr = learning_rate
    decay = lr_decay
    opt = optimizer_name

    if job_dir is None:
        job_dir = './tmp/'

    if job_id is None:
        ctime = time.ctime().split()
        time_str = ctime[4]+ctime[1]+ctime[2]+"_"+''.join(ctime[3].split(":")[0:2])
        job_id = time_str
        checkpoint_filename = 'mnist_autoencoder_checkpoint_{}.hdf5'.format(job_id)
    else:
        checkpoint_filename = '{}.hdf5'.format(job_id)

    if use_transposed_conv is True:
        print('--> use_transposed_conv is True!!!\n')
        job_id = 'transposed_conv_' + job_id


    ############ Encoder ###############
    x = Input(shape=original_img_size) # reshape to: (100, 28, 28, 1)

    conv1 = conv_block(x, n_filters, kernel_size)
    conv1 = conv_block(conv1, n_filters, kernel_size)
    conv1 = pool_layer(conv1, method=pool_method, pool_size=(2,2), padding='same')

    conv2 = conv_block(conv1, n_filters*2, kernel_size)
    conv2 = conv_block(conv2, n_filters*2, kernel_size)
    conv2 = pool_layer(conv2, method=pool_method, pool_size=(2,2), padding='same')

    conv3 = conv_block(conv2, n_filters*4, kernel_size)
    conv3 = conv_block(conv3, n_filters*4, kernel_size)
    encoded = pool_layer(conv3, method=pool_method, pool_size=(2,2), padding='same')

    # End of encoder. The compressed representation is (4, 4, 8)

    conv4 = conv_block(encoded, n_filters*4, kernel_size, use_transposed_conv=use_transposed_conv)
    conv4 = conv_block(conv4, n_filters*4, kernel_size, use_transposed_conv=use_transposed_conv)
    conv4 = UpSampling2D((2, 2))(conv4)

    conv5 = conv_block(conv4, n_filters*2, kernel_size, use_transposed_conv=use_transposed_conv)
    conv5 = conv_block(conv5, n_filters*2, kernel_size, use_transposed_conv=use_transposed_conv)
    conv5 = UpSampling2D((2, 2))(conv5)

    conv6 = conv_block(conv5, n_filters, kernel_size, use_transposed_conv=use_transposed_conv)
    conv6 = conv_block(conv6, n_filters, kernel_size, use_transposed_conv=use_transposed_conv)
    conv6 = UpSampling2D((2, 2))(conv6)

    decoded = conv_block(conv6, 1, kernel_size=kernel_size, activation='sigmoid', padding='same',
                         batch_norm=False, use_transposed_conv=use_transposed_conv) # the activation here is sigmoid b/c the pixel values are bounded b/w 0-1, and there are lots of 0s
    decoded = Cropping2D(cropping=((2, 2), (2, 2)))(decoded) # crop 2 on each side of the img to get 28x28

    # Put all layers together into a model graph
    autoencoder = Model(x, decoded)

    ######### End of decoder ###################
    ######### Now config models for training and logging ##########

    if opt =='adam':
        optimizer = Adam(lr=lr, decay=decay)
    elif opt =='sgd':
        optimizer = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=True)

    autoencoder.compile(optimizer=optimizer, loss = loss, metrics = [score_metric])
    autoencoder.summary()


    # data from MNIST digits
    (x_train, _), (x_test, y_test) = mnist.load_data()

    # reshape data to (data_size, n_pix, n_pix, n_channels)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    print('x_train.shape:', x_train.shape)

    callbacks = [EarlyStopping(monitor='val_loss',patience=5,verbose=2, mode='min', min_delta=0.0005),
                 ModelCheckpoint(checkpoint_filename, monitor='val_loss', verbose=2, save_best_only=True),
                 TensorBoard(log_dir=job_dir)]

    history = autoencoder.fit(x=x_train, y=x_train, shuffle=True, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks,
                        verbose=2, validation_data=(x_test, x_test))

    test_score = autoencoder.evaluate(x_test, x_test, verbose=0, batch_size=batch_size)
    print('Final test score:', test_score)

    if score_metric == 'mae':
        history_key_validation = 'val_mean_absolute_error'
        history_key_train = 'mean_absolute_error'
    if score_metric == 'mse':
        history_key_validation = 'val_mean_squared_error'
        history_key_train = 'mean_squared_error'

    validation_history = history.history[history_key_validation]
    training_history = history.history[history_key_train]


    # Save model to gs
    if 'gs://' in job_dir:
        # Save model hdf5 to google storage
        with file_io.FileIO(checkpoint_filename, mode='rb') as input_f:
            with file_io.FileIO(job_dir + checkpoint_filename, mode='w') as output_f:
                output_f.write(input_f.read())


    #%% Plot a learning curve
    fig_name = 'lr_{}.pdf'.format(job_id) #
    if 'gs://' not in job_dir:
        fig_name = job_dir + fig_name

    f, axes = plt.subplots(2, sharex=True, figsize=(8,7))
    axes[0].plot(training_history)
    axes[0].set_ylabel('Training score ({})'.format(score_metric))
    axes[0].set_title('Final test score ({0}) = {1:2.4f}\n LR={2}, decay={3}, optimizer={4}, pool_method={5}\n \
                    use_transposed_conv={6}, loss={7}, batch_norm_before_activation={8}'.format(score_metric,
                    test_score[1], lr, decay, opt, pool_method, use_transposed_conv, loss, do_batch_norm_before_activation), fontsize=9)

    axes[1].plot(validation_history)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Validation score ({})'.format(score_metric))
    #f.suptitle('Config file :{}'.format(train_config_file), fontsize=10)
    f.subplots_adjust(hspace=0.05)
    f.savefig(fig_name)

    if 'gs://' in job_dir:
        #Save figure to GS
        with file_io.FileIO(fig_name, mode='rb') as input_f:
            with file_io.FileIO(job_dir + fig_name, mode='w') as output_f:
                output_f.write(input_f.read())


    #%% Sample a few test images and compare with reconstructed ones
    n_imgs_to_show = 30
    x_test_sub = np.random.permutation(x_test)[0:n_imgs_to_show]

    reconstructed_test = autoencoder.predict(x_test_sub, batch_size=n_imgs_to_show)
    #reconstructed_train = autoencoder.predict(x_train[0:10000].reshape((10000, 28, 28, 1)), batch_size=batch_size)

    # plot reconstructed images and compare
    fig_name = 'compare_{}.pdf'.format(job_id)
    if 'gs://' not in job_dir:
        fig_name = job_dir + fig_name

    n_rows = 3 # split orignal images into 2 rows
    n_cols = n_imgs_to_show//n_rows
    f, axes = plt.subplots(n_rows*2, n_cols, sharey=True, figsize=(10,10))

    for i in range(n_imgs_to_show):
        axes[i//n_cols * 2, i % n_cols].imshow(x_test_sub[i,:,:,0])
        axes[i//n_cols * 2, 0].set_ylabel('Original')
        axes[i//n_cols * 2 +1, i % n_cols].imshow(reconstructed_test[i,:,:,0])
        axes[i//n_cols * 2 +1, 0].set_ylabel('Reconstructed')
    f.savefig(fig_name)

    if 'gs://' in job_dir:
        #Save figure to GS
        with file_io.FileIO(fig_name, mode='rb') as input_f:
            with file_io.FileIO(job_dir + fig_name, mode='w') as output_f:
                output_f.write(input_f.read())


#%%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # This argument is required by GC
    parser.add_argument(
      '--job_dir', help='GCS location to write checkpoints and export models', default=None)

    parser.add_argument('--job_id', help='Job ID to tag models', default=None)

    parser.add_argument('--use_transposed_conv', help='Use "deconv" layers or transposed conv layers for "deconv"',
                        action='store_true', default=False)

    parser.add_argument('--score_metric', help='Metric for scoring: mse, mae, binary_entropy, etc.',
                        default='mse')

    parser.add_argument('--loss', help='Loss function: mse, mae, binary_crossentropy, etc.',
                        default='binary_crossentropy')

    parser.add_argument('-lr', '--learning_rate', help='learning rate',
                        default=0.001, type=float)

    parser.add_argument('--lr_decay', help='Learning rate decay (e.g., linear decay for Adma)',
                        default=0.001, type=float)

    parser.add_argument('-opt','--optimizer_name', help='optimizer function name',
                        default='adam')

    parser.add_argument('--n_epochs', help='Number of epochs',
                        default=100, type=int)

    parser.add_argument('--patience', help='Number of epochs to wait before early stopping',
                        default=5, type=int)

    parser.add_argument('--pool_method', help='Pooling method, either "max" or "average"',
                        default="max")

    args = parser.parse_args()
    arguments = args.__dict__

    train(**arguments)
