import argparse
import json
import os
import sys
from contextlib import nullcontext

# no more TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

import numpy as np
import tensorflow as tf
import neptune.new as neptune

from gl_rep.data_loaders import airq_data_loader, physionet_data_loader
from models.cdsvae import Encoder, Decoder, CDSVAE
from models.Predictor_daily_rain import Predictor

tf.get_logger().setLevel(logging.ERROR)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

print_losses = lambda type, loss, nll, kld_f, kld_z, mi_fz: \
    "{} loss = {:.3f} \t NLL = {:.3f} \t KL_f = {:.3f} \t KL_z = {:.3f} \t MI_fz = {:.3f}".\
        format(type, loss, nll, kld_f, kld_z, mi_fz)

window_size = 1 * 24


def run_epoch(args, model, dataset, data_type, opt=None, train=False, vars=None):
    model.dataset_size = len(dataset)
    LOSSES = []
    for i, batch in dataset.enumerate():
        x_seq, mask_seq, x_lens = batch[0], batch[1], batch[2]

        with tf.GradientTape() if train else nullcontext() as gen_tape:
            losses = model.compute_loss(x_seq, m_mask=mask_seq, x_len=x_lens, return_parts=True)
        if train:
            gradients_of_generator = gen_tape.gradient(losses[0], vars)
            opt.apply_gradients(zip(gradients_of_generator, vars))

        losses = np.asarray([loss.numpy() for loss in losses])
        if args.neptune_run:
            tr_va_str = 'train' if train else 'valid'
            args.neptune_run[f'{tr_va_str}/loss'].log(losses[0])
            args.neptune_run[f'{tr_va_str}/nll'].log(losses[1])
            args.neptune_run[f'{tr_va_str}/kld_f'].log(losses[2])
            args.neptune_run[f'{tr_va_str}/kld_z'].log(losses[3])
            args.neptune_run[f'{tr_va_str}/mi_fz'].log(losses[4])

        LOSSES.append(losses)
    LOSSES = np.stack(LOSSES)
    return np.mean(LOSSES, axis=0)


def run_epoch_dt(model, dataset, glr_model, optimizer=None, label_blocks=None, train=False , trainable_vars=None):
    "Training epoch for training the classifier"
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    mae_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    epoch_loss, epoch_acc, epoch_auroc = [], [], []

    if label_blocks is None:
        all_labels_blocks = tf.concat([b[3][:, :, 1] for b in dataset], 0)
        all_labels_blocks = tf.split(all_labels_blocks, num_or_size_splits=all_labels_blocks.shape[1] // window_size, axis=1)
        label_blocks = tf.stack([tf.math.reduce_sum(block, axis=1) for block in all_labels_blocks], axis=-1)

    b_start = 0
    for batch_i, batch in dataset.enumerate():
        x_seq = batch[0]
        mask_seq, x_lens = batch[1], batch[2]

        labels = label_blocks[b_start:b_start+len(x_seq)]
        b_start += len(x_seq)
        labels = tf.where(tf.math.is_nan(labels), tf.zeros_like(labels), labels)

        _, _, _, z_g, _, _, _, z_t, _, _, _, _, _, _ = glr_model(x_seq, mask_seq)
        lens = x_lens//glr_model.window_size


        if train:
            with tf.GradientTape() as gen_tape:
                predictions = model(z_t, z_g, lens)
                loss = tf.abs(tf.subtract(tf.cast(labels, dtype=tf.float32),tf.cast(predictions, dtype=tf.float32)))
                loss_weight = tf.cast(tf.where(labels==0, 1.0, 1.), dtype=tf.float32)
                loss = tf.reduce_mean(loss*loss_weight)
            grads = gen_tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
        else:
            predictions = model(z_t, z_g, lens)
        epoch_loss.append(mae_loss(labels, predictions).numpy().mean())
    return np.mean(epoch_loss)

def train_cdsvae(args, model, trainset, validset, file_name, lr=1e-4, n_epochs=2):
    opt = tf.keras.optimizers.Adam(lr)
    _ = tf.compat.v1.train.get_or_create_global_step()
    vars = model.get_trainable_vars()

    for epoch in range(n_epochs+1):
        ep_losses = run_epoch(args, model, trainset, args.data, train=True, opt=opt, vars=vars)
        ep_loss, ep_nll, ep_kld_f, ep_kld_z, ep_mi_fz = ep_losses

        # print losses during training
        print('='*30)
        print('Epoch {}, (Learning rate: {:.5f})'.format(epoch, lr))
        print('{}'.format(print_losses('Training', ep_loss, ep_nll, ep_kld_f, ep_kld_z, ep_mi_fz)))

        ep_loss, ep_nll, ep_kld_f, ep_kld_z, ep_mi_fz = run_epoch(args, model, validset, args.data)
        print('{}'.format(print_losses('Validation', ep_loss, ep_nll, ep_kld_f, ep_kld_z, ep_mi_fz)))

        # save model
        model.save_weights(f'{args.ckpt}{file_name}')


def block_labels(dataset):
    all_labels_blocks = tf.concat([b[3][:, :, 1] for b in dataset], 0)
    all_labels_blocks = tf.split(all_labels_blocks, num_or_size_splits=all_labels_blocks.shape[1] // window_size, axis=1)
    all_labels_blocks = tf.stack([tf.math.reduce_sum(block, axis=1) for block in all_labels_blocks], axis=-1)
    return all_labels_blocks


def main(args):
    is_continue = False  # Continue training an existing checkpoint
    file_name = 'cdsvae%d_%d_%s' % (args.f_dim, args.z_dim, args.data)

    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]

    if args.data == 'air_quality':
        n_epochs = 250
        trainset, validset, testset, normalization_specs = airq_data_loader(normalize="mean_zero")
    elif args.data == 'physionet':
        n_epochs = 200
        trainset, validset, testset, normalization_specs = physionet_data_loader(normalize="mean_zero")

    if args.train:
        print(f'Training CDSVAE model on {args.data} saving to file {file_name}')
        print(args)

    # Create Neptune
    args.neptune_run = None
    if args.neptune:
        run = neptune.init(
            project="azencot-group/Contrastive-TimeSeries",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNzRiNTcyYy1jNWY1LTRjNTAtOTc3OS04NDE2MjllMjVjMzcifQ==",
        )  # your credentials
        run['config/hyperparameters'] = vars(args)
        run['data'] = args.data
        run['f_dim'] = args.f_dim
        run['z_dim'] = args.z_dim
        run['weight_rec'] = args.weight_rec
        run['weight_f'] = args.weight_f
        run['weight_z'] = args.weight_z
        run['weight_mi'] = args.weight_mi
        run["sys/tags"].add(['cdsvae, ilan'])
        args.neptune_run = run

    # Create the representation learning models
    encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
    decoder = Decoder(output_size=configs["feature_size"],
                      output_length=configs["window_size"],
                      hidden_sizes=configs["cdsvae_decoder_size"])
    rep_model = CDSVAE(encoder=encoder, decoder=decoder, configs=configs, args=args)


    # Train the CDSVAE baseline
    if args.train:
        if not os.path.exists(f'{args.ckpt}'):
            os.mkdir(f'{args.ckpt}')

        if is_continue:
            rep_model.load_weights(f'{args.ckpt}{file_name}')
        train_cdsvae(args, rep_model, trainset, validset, lr=1e-3, n_epochs=n_epochs, file_name=file_name)

    # Report test performance
    rep_model.load_weights(f'{args.ckpt}{file_name}')
    test_loss, test_nll, test_kld_f, test_kld_z, test_mi_fz = run_epoch(rep_model, testset, args.data)
    print(f'\nCDSVAE performance on {args.data} data')
    print('{}'.format(print_losses('Test', test_loss, test_nll, test_kld_f, test_kld_z, test_mi_fz)))


    ##### ----- finished training our model, now train downstream task ----- #####

    test_loss = []
    for cv in range(3):
        model = Predictor([32, 8])
        trainset, validset, testset, normalization_specs = airq_data_loader(normalize='mean_zero')
        label_blocks_train = block_labels(trainset)
        label_blocks_valid = block_labels(validset)
        label_blocks_test = block_labels(testset)
        n_epochs = 50
        lr = 1e-4

        with open('configs.json') as config_file:
            configs = json.load(config_file)['air_quality']

        model(tf.random.normal(shape=(5, 10, args.z_dim), dtype=tf.float32),
              tf.random.normal(shape=(5, args.f_dim), dtype=tf.float32),
              x_lens=tf.ones(shape=(5,)) * 10)
        optimizer = tf.keras.optimizers.Adam(lr)
        trainable_vars = model.trainable_variables
        losses_train, acc_train, auroc_train = [], [], []
        losses_val, acc_val, auroc_val = [], [], []
        for epoch in range(n_epochs+1):
            epoch_loss = run_epoch(model, trainset, rep_model, optimizer=optimizer, label_blocks = label_blocks_train,
                                                           train=True, trainable_vars=trainable_vars)
            if epoch and epoch % 10 == 0:
                print('=' * 30)
                print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
                losses_train.append(epoch_loss)

                epoch_loss = run_epoch(model, validset, rep_model, label_blocks = label_blocks_valid, train=False)
                losses_val.append(epoch_loss)
                te_loss = run_epoch(model, testset, rep_model, label_blocks=label_blocks_test, train=False)

                print("Training loss = %.3f" % (epoch_loss))
                print("Validation loss = %.3f" % (epoch_loss))
                print('Test loss =  %.3f'% (te_loss))

                if args.neptune_run:
                    args.neptune_run[f'train/loss'].log(epoch_loss)
                    args.neptune_run[f'val/loss'].log(epoch_loss)
                    args.neptune_run[f'test/loss'].log(te_loss)
        test_loss.append(run_epoch(model, testset, rep_model, label_blocks = label_blocks_test, train=False))
    print("\n\n Final performance \t loss = %.3f +- %.3f" % (np.mean(test_loss), np.std(test_loss)))
    if args.neptune_run:
        args.neptune_run[f'test/final_loss_m'].log(np.mean(test_loss))
        args.neptune_run[f'test/final_loss_std'].log(np.std(test_loss))
    model.save_weights(file_name)


    if args.neptune_run:
        args.neptune_run.stop()
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--ckpt', type=str, default='./ckpt/')
    parser.add_argument('--data', type=str, default='air_quality', help="dataset to use")
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    parser.add_argument('--rep_size', type=int, default=8, help="Size of the representation vectors")
    parser.add_argument('--f_dim', type=int, default=12, help="Size of static vector")
    parser.add_argument('--z_dim', type=int, default=4, help="Size of dynamic vector")
    parser.add_argument('--weight_rec', type=float, default=15., help='weighting on general reconstruction')
    parser.add_argument('--weight_f', type=float, default=7, help='weighting on KL to prior, content vector')
    parser.add_argument('--weight_z', type=float, default=1, help='weighting on KL to prior, motion vector')
    parser.add_argument('--weight_mi', default=2.5, type=float, help='weighting on mutual information loss')
    parser.add_argument('--weight_f_aug', default=.1, type=float, help='weighting on static contrastive loss')
    parser.add_argument('--weight_z_aug', default=.1, type=float, help='weighting on dynamic contrastive loss')
    parser.add_argument('--note', default='sample', type=str, help='appx note')
    parser.add_argument('--seed', default=1948, type=int, help='random seed')
    parser.add_argument('--neptune', default=False, type=bool, help='activate neptune tracking')

    args = parser.parse_args()
    main(args)
