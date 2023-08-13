# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Daily rain estimation experiment with the Air Quality dataset
Using a simple MLP model, we estimate the average daily rain from
the local representations of the time series that are learned
in an unsupervised manner.
"""


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
sns.set()
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
#
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from gl_rep.models import EncoderLocal, EncoderGlobal, WindowDecoder
from gl_rep.glr import GLR
from gl_rep.data_loaders import airq_data_loader
import neptune.new as neptune

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

window_size = 1 * 24


def main(args):
    with open('configs.json') as config_file:
        configs = json.load(config_file)['air_quality']

    # Create Neptune
    args.neptune_run = None
    if args.neptune:
        run = neptune.init_run(
            project="azencot-group/Contrastive-TimeSeries",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNzRiNTcyYy1jNWY1LTRjNTAtOTc3OS04NDE2MjllMjVjMzcifQ==",
        )  # your credentials
        run['config/hyperparameters'] = vars(args)
        run['f_dim'] = args.f_dim
        run['z_dim'] = args.z_dim
        run['weight_rec'] = args.weight_rec
        run['weight_f'] = args.weight_f
        run['weight_z'] = args.weight_z
        run['weight_mi'] = args.weight_mi
        run['weight_f_aug'] = args.weight_f_aug
        run['weight_z_aug'] = args.weight_z_aug
        run['ce_epoch'] = args.ce_epoch
        run["sys/tags"].add(['ours4-DR downstream, ilan, small batch'])
        args.neptune_run = run

    mode = args.mode
    if args.train:
        test_loss  = []
        for cv in range(3):
            model = Predictor([32, 8])
            trainset, validset, testset, normalization_specs = airq_data_loader(normalize='mean_zero')
            label_blocks_train = block_labels(trainset)
            label_blocks_valid = block_labels(validset)
            label_blocks_test = block_labels(testset)
            if mode == 'supervised':
                n_epochs = 200
                lr = 1e-3
                file_name = './ckpt/e2e_rain_prediction'
                rep_model_file = 'End to end'
                rep_model = None
            else:
                n_epochs = 50
                lr = 1e-4
                # lr = 1e-3
                if mode == 'glr':
                    file_name = './ckpt/glr_rain_predictor'
                    rep_model_file = './ckpt/glr_air_quality_lambda%.1f'%args.lamda
                    zt_encoder = EncoderLocal(zl_size=configs["zl_size"],
                                              hidden_sizes=configs["glr_local_encoder_size"])
                    zg_encoder = EncoderGlobal(zg_size=configs["zg_size"],
                                               hidden_sizes=configs["glr_global_encoder_size"])
                    dec = WindowDecoder(output_size=configs["feature_size"], output_length=configs["window_size"],
                                        hidden_sizes=configs["glr_decoder_size"])
                    rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, decoder=dec,
                                    window_size=configs["window_size"], time_length=configs["t_len"],
                                    data_dim=configs["feature_size"], kernel_scales=configs["kernel_scales"],
                                    kernel=configs["kernels"], beta=configs["beta"], M=configs["mc_samples"], sigma=.5,
                                    lamda=args.lamda, length_scale=configs["length_scale"], p=15)
                elif mode == 'gpvae':
                    file_name = './ckpt/gpvae%d_rain_predictor'%args.rep_size
                    rep_model_file = './ckpt/gpvae%d_air_quality'%args.rep_size
                    encoder = EncoderLocal(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
                    decoder = Decoder(output_size=configs["feature_size"],
                                      output_length=configs["window_size"],
                                      hidden_sizes=configs["baseline_decoder_size"])
                    rep_model = GPVAE(encoder, decoder, time_length=configs["t_len"], data_dim=configs["feature_size"],
                                      window_size=configs["window_size"], kernel=['cauchy'], beta=1., M=1,
                                      sigma=1.0, length_scale=2.0, kernel_scales=4, p=100)
                elif mode == 'vae':
                    file_name = './ckpt/vae%d_rain_predictor'%args.rep_size
                    rep_model_file = './ckpt/vae%d_air_quality'%args.rep_size
                    encoder = Encoder(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
                    decoder = Decoder(output_size=configs["feature_size"],
                                      output_length=configs["window_size"],
                                      hidden_sizes=configs["baseline_decoder_size"])
                    rep_model = VAE(encoder=encoder, decoder=decoder, data_dim=configs["feature_size"], beta=1.,
                                    M=configs["mc_samples"], sample_len=configs["t_len"])
                elif mode == 'cdsvae':
                    from train_cdsvae import CDSVAE, Encoder, Decoder
                    file_name = './ckpt/cdsvae%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                    rep_model_file = './ckpt/cdsvae%d_%d_air_quality' % (args.f_dim, args.z_dim)
                    encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                    decoder = Decoder(output_size=configs["feature_size"],
                                      output_length=configs["window_size"],
                                      hidden_sizes=configs["cdsvae_decoder_size"])
                    rep_model = CDSVAE(encoder=encoder, decoder=decoder, configs=configs, args=args)
                elif mode == 'ours':
                    from ours import OURS, Encoder, Decoder
                    file_name = './ckpt/ours%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                    rep_model_file = './ckpt/ours%d_%d_air_quality' % (args.f_dim, args.z_dim)
                    encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                    decoder = Decoder(output_size=configs["feature_size"],
                                      output_length=configs["window_size"],
                                      hidden_sizes=configs["cdsvae_decoder_size"])
                    rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
                elif mode == 'ours2':
                    from ours2 import OURS, Encoder, Decoder
                    file_name = './ckpt/ours2%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                    rep_model_file = './ckpt/ours2%d_%d_air_quality' % (args.f_dim, args.z_dim)
                    encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                    decoder = Decoder(output_size=configs["feature_size"],
                                      output_length=configs["window_size"],
                                      hidden_sizes=configs["cdsvae_decoder_size"])
                    rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
                elif mode == 'ours3':
                    from ours3 import OURS, Encoder, Decoder
                    file_name = './ckpt/ours3%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                    # rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                    #          (args.f_dim, args.z_dim, args.ce_epoch, args.weight_f_aug,
                    #           args.weight_mi, args.weight_rec, args.note)
                    rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wf=%f_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                             (args.f_dim, args.z_dim, args.weight_f, args.ce_epoch, args.weight_f_aug,
                              args.weight_mi, args.weight_rec, args.note)
                    encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                    decoder = Decoder(output_size=configs["feature_size"],
                                      output_length=configs["window_size"],
                                      hidden_sizes=configs["cdsvae_decoder_size"])
                    rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
                elif mode == 'ours4':
                    from ours4 import OURS, Encoder, Decoder
                    file_name = './ckpt/ours3%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                    # rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                    #          (args.f_dim, args.z_dim, args.ce_epoch, args.weight_f_aug,
                    #           args.weight_mi, args.weight_rec, args.note)
                    rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wf=%f_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                             (args.f_dim, args.z_dim, args.weight_f, args.ce_epoch, args.weight_f_aug,
                              args.weight_mi, args.weight_rec, args.note)
                    encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                    decoder = Decoder(output_size=configs["feature_size"],
                                      output_length=configs["window_size"],
                                      hidden_sizes=configs["cdsvae_decoder_size"])
                    rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
                rep_model.load_weights(rep_model_file)

            print(f'Training {rep_model_file}, saving to {file_name}')
            if mode == 'glr':
                model(tf.random.normal(shape=(5, 10, zt_encoder.zl_size), dtype=tf.float32),
                      tf.random.normal(shape=(5, zt_encoder.zl_size), dtype=tf.float32),
                      x_lens=tf.ones(shape=(5,))*10)
                optimizer = tf.keras.optimizers.Adam(lr)
            elif mode in ['cdsvae', 'ours', 'ours2', 'ours3', 'ours4']:
                model(tf.random.normal(shape=(5, 10, args.z_dim), dtype=tf.float32),
                      tf.random.normal(shape=(5, args.f_dim), dtype=tf.float32),
                      x_lens=tf.ones(shape=(5,)) * 10)
                optimizer = tf.keras.optimizers.Adam(lr)
            elif mode == 'supervised':
                model(tf.random.normal(shape=(5, 10, configs["feature_size"]), dtype=tf.float32), None,
                      x_lens=tf.ones(shape=(5,))*10)
                optimizer = tf.keras.optimizers.Adam(lr)
            else:
                model(tf.random.normal(shape=(5, 10, encoder.zl_size), dtype=tf.float32), None,
                      x_lens=tf.ones(shape=(5,))*10)
                optimizer = tf.keras.optimizers.Adam(lr)
            trainable_vars = model.trainable_variables
            losses_train, acc_train, auroc_train = [], [], []
            losses_val, acc_val, auroc_val = [], [], []
            for epoch in range(n_epochs+1):
                epoch_loss = run_epoch(model, trainset, rep_model, optimizer=optimizer, label_blocks = label_blocks_train,
                                                               train=True, trainable_vars=trainable_vars, mode=mode)
                if epoch and epoch % 10 == 0:
                    print('=' * 30)
                    print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
                    losses_train.append(epoch_loss)
                    if args.neptune_run:
                        args.neptune_run[f'train/loss'].log(epoch_loss)
                    print("Training loss = %.3f" % (epoch_loss))
                    epoch_loss = run_epoch(model, validset, rep_model, label_blocks = label_blocks_valid, train=False, mode=mode)
                    losses_val.append(epoch_loss)
                    if args.neptune_run:
                        args.neptune_run[f'val/loss'].log(epoch_loss)
                    print("Validation loss = %.3f" % (epoch_loss))
                    te_loss = run_epoch(model, testset, rep_model, label_blocks=label_blocks_test, train=False, mode=mode)
                    print('Test loss =  %.3f'% (te_loss))
                    if args.neptune_run:
                        args.neptune_run[f'test/loss'].log(te_loss)
            test_loss.append(run_epoch(model, testset, rep_model, label_blocks = label_blocks_test, train=False, mode=mode))
        print("\n\n Final performance \t loss = %.3f +- %.3f" % (np.mean(test_loss), np.std(test_loss)))
        if args.neptune_run:
            args.neptune_run[f'test/final_loss'].log(np.mean(test_loss))
        plt.figure()
        plt.plot(losses_train, label='Train loss')
        plt.plot(losses_val, label='Validation loss')
        plt.legend()
        plt.savefig('./plots/evaluations/rain_prediction_loss_%s.pdf' % (mode))
        model.save_weights(file_name)

    else:
        trainset, validset, testset, normalization_specs = airq_data_loader(normalize='mean_zero')
        if mode == 'supervised':
            file_name = './ckpt/e2e_rain_prediction'
            rep_model = None
        else:
            if mode == 'glr':
                file_name = './ckpt/glr_rain_predictor'
                rep_model_file = './ckpt/glr_air_quality_lambda%.1f' %(args.lamda)
                zt_encoder = EncoderLocal(zl_size=configs["zl_size"],
                                          hidden_sizes=configs["glr_local_encoder_size"])
                zg_encoder = EncoderGlobal(zg_size=configs["zg_size"],
                                           hidden_sizes=configs["glr_global_encoder_size"])
                dec = WindowDecoder(output_size=configs["feature_size"], output_length=configs["window_size"],
                                    hidden_sizes=configs["glr_decoder_size"])
                rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, decoder=dec,
                                window_size=configs["window_size"], time_length=configs["t_len"],
                                data_dim=configs["feature_size"], kernel_scales=configs["kernel_scales"],
                                kernel=configs["kernels"], beta=configs["beta"], M=configs["mc_samples"], sigma=.5,
                                lamda=args.lamda, length_scale=configs["length_scale"], p=15)
            elif mode == 'gpvae':
                from baselines.gpvae import GPVAE, Decoder
                file_name = './ckpt/gpvae%d_rain_predictor'%args.rep_size
                rep_model_file = './ckpt/gpvae%d_air_quality'%args.rep_size
                encoder = EncoderLocal(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
                decoder = Decoder(output_size=configs["feature_size"],
                                  output_length=configs["window_size"],
                                  hidden_sizes=configs["baseline_decoder_size"])
                rep_model = GPVAE(encoder, decoder, time_length=configs["t_len"], data_dim=configs["feature_size"],
                                  window_size=configs["window_size"], kernel=['cauchy'], beta=1., M=1,
                                  sigma=1.0, length_scale=2.0, kernel_scales=4, p=100)
            elif mode == 'vae':
                from baselines.vae import VAE, Encoder, Decoder
                file_name = './ckpt/vae%d_rain_predictor'%args.rep_size
                rep_model_file = './ckpt/vae%d_air_quality'%args.rep_size
                encoder = Encoder(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
                decoder = Decoder(output_size=configs["feature_size"],
                                  output_length=configs["window_size"],
                                  hidden_sizes=configs["baseline_decoder_size"])
                rep_model = VAE(encoder=encoder, decoder=decoder, data_dim=configs["feature_size"], beta=1.,
                                M=configs["mc_samples"], sample_len=configs["t_len"])
            elif mode == 'cdsvae':
                from train_cdsvae import CDSVAE, Encoder, Decoder
                file_name = './ckpt/cdsvae%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                rep_model_file = './ckpt/cdsvae%d_%d_air_quality' % (args.f_dim, args.z_dim)
                encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                decoder = Decoder(output_size=configs["feature_size"],
                                  output_length=configs["window_size"],
                                  hidden_sizes=configs["cdsvae_decoder_size"])
                rep_model = CDSVAE(encoder=encoder, decoder=decoder, configs=configs, args=args)
            elif mode == 'ours':
                from ours import OURS, Encoder, Decoder
                file_name = './ckpt/ours%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                rep_model_file = './ckpt/ours%d_%d_air_quality' % (args.f_dim, args.z_dim)
                encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                decoder = Decoder(output_size=configs["feature_size"],
                                  output_length=configs["window_size"],
                                  hidden_sizes=configs["cdsvae_decoder_size"])
                rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
            elif mode == 'ours2':
                from ours2 import OURS, Encoder, Decoder
                file_name = './ckpt/ours2%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                rep_model_file = './ckpt/ours2%d_%d_air_quality' % (args.f_dim, args.z_dim)
                encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                decoder = Decoder(output_size=configs["feature_size"],
                                  output_length=configs["window_size"],
                                  hidden_sizes=configs["cdsvae_decoder_size"])
                rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
            elif mode == 'ours3':
                from ours2 import OURS, Encoder, Decoder
                file_name = './ckpt/ours3%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                # rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                #              (args.f_dim, args.z_dim, args.ce_epoch, args.weight_f_aug,
                #               args.weight_mi, args.weight_rec, args.note)
                rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wf=%f_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                                 (args.f_dim, args.z_dim, args.weight_f, args.ce_epoch, args.weight_f_aug,
                                  args.weight_mi, args.weight_rec, args.note)
                encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                decoder = Decoder(output_size=configs["feature_size"],
                                  output_length=configs["window_size"],
                                  hidden_sizes=configs["cdsvae_decoder_size"])
                rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
            elif mode == 'ours4':
                from ours4 import OURS, Encoder, Decoder
                file_name = './ckpt/ours3%d_%d_rain_predictor' % (args.f_dim, args.z_dim)
                # rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                #              (args.f_dim, args.z_dim, args.ce_epoch, args.weight_f_aug,
                #               args.weight_mi, args.weight_rec, args.note)
                rep_model_file = './ckpt/ours3_fdim=%d_z_dim=%d_wf=%f_wuep=%d_waug=%f_wmi=%f_wrec=%f_air_quality_note=%s' % \
                                 (args.f_dim, args.z_dim, args.weight_f, args.ce_epoch, args.weight_f_aug,
                                  args.weight_mi, args.weight_rec, args.note)
                encoder = Encoder(hidden_sizes=configs["cdsvae_encoder_size"])
                decoder = Decoder(output_size=configs["feature_size"],
                                  output_length=configs["window_size"],
                                  hidden_sizes=configs["cdsvae_decoder_size"])
                rep_model = OURS(encoder=encoder, decoder=decoder, configs=configs, args=args)
            rep_model.load_weights(rep_model_file)

        model = Predictor([32, 8])
        if not mode == 'supervised':
            rep_model.load_weights(rep_model_file)#.expect_partial()
        model.load_weights(file_name).expect_partial()

        # Plot the estimated daily rain for a random sample
        for batch in testset:
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            rnd_sample = tf.math.argmax(x_lens)
            all_labels = batch[3][:, :, 1]

            if mode == 'glr':
                global_sample_len = int(x_seq.shape[1] * 0.3)
                rnd_t_g = np.random.randint(0, x_seq.shape[1] - global_sample_len)
                z_g = rep_model.global_encoder(x_seq[:, rnd_t_g:rnd_t_g + global_sample_len, :],
                                          mask=mask_seq[:, rnd_t_g:rnd_t_g + global_sample_len, :]).sample()
                pz_t = rep_model.local_encoder(x_seq,mask=mask_seq, window_size=window_size)
                z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
                pred = model(z_t, z_g, x_lens)[rnd_sample]
                all_labels_blocks = tf.split(all_labels, num_or_size_splits=all_labels.shape[1] // window_size, axis=1)
                labels = tf.stack([tf.math.reduce_sum(block, axis=1) for block in all_labels_blocks], axis=-1)[rnd_sample]
            elif mode == 'gpvae':
                z_t, _ = rep_model.encode(x_seq, mask_seq)
                z_g = None
                pred = model(z_t, z_g, x_lens)[rnd_sample]
                all_labels_blocks = tf.split(all_labels, num_or_size_splits=all_labels.shape[1] // window_size, axis=1)
                labels = tf.stack([tf.experimental.numpy.nanmean(block, axis=1) for block in all_labels_blocks], axis=-1)[rnd_sample]
            elif mode == 'vae':
                z_t = rep_model.encoder(x_seq, mask_seq, rep_model.window_size).sample()
                z_g = None
                pred = model(z_t, z_g, x_lens)[rnd_sample]
                all_labels_blocks = tf.split(all_labels, num_or_size_splits=all_labels.shape[1] // window_size, axis=1)
                labels = tf.stack([tf.experimental.numpy.nanmean(block, axis=1) for block in all_labels_blocks], axis=-1)[rnd_sample]
            elif mode in ['cdsvae', 'ours', 'ours2', 'ours3', 'ours4']:
                _, _, _, z_g, _, _, _, z_t, _, _, _, _, _, _ = rep_model(x_seq, mask_seq)
                pred = model(z_t, z_g, x_lens)[rnd_sample]
                all_labels_blocks = tf.split(all_labels, num_or_size_splits=all_labels.shape[1] // window_size, axis=1)
                labels = tf.stack([tf.experimental.numpy.nanmean(block, axis=1) for block in all_labels_blocks], axis=-1)[rnd_sample]
            elif mode == 'supervised':
                pred = model(x_seq, None, x_lens)[rnd_sample]
                labels = all_labels[rnd_sample]
            plt.figure(figsize=(10,4))
            plt.plot(labels, label='Average Daily Rain')
            plt.plot(pred, label='Estimated Daily Rain')
            plt.legend()
            plt.savefig('./plots/evaluations/estimated_rain_%s.pdf'%mode)
            break
    if args.neptune_run:
        args.neptune_run.stop()


class Predictor(tf.keras.Model):
    """Simple classifier layer to classify the subgroup of data

    Args:
        fc_sizes: Hidden size of the predictor MLP
    """
    def __init__(self, fc_sizes):
        super(Predictor, self).__init__()
        self.fc_sizes = fc_sizes
        self.fc = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                                       for h in fc_sizes])
        self.prob = tf.keras.layers.Dense(1, activation=tf.nn.relu, dtype=tf.float32)

    def __call__(self, local_encs, global_encs, x_lens):
        if not global_encs is None:
            h = tf.concat([local_encs, tf.tile(tf.expand_dims(global_encs, axis=1), [1, local_encs.shape[1], 1])], axis=-1)
            h = tf.keras.layers.BatchNormalization()(h)
        else:
            h = local_encs
        logits = (self.fc(h))
        probs = tf.keras.layers.Dropout(rate=0.3)(self.prob(logits))
        return probs[...,0]


def run_epoch(model, dataset, glr_model, optimizer=None, label_blocks=None, train=False , trainable_vars=None, mode='glr'):
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

        all_labels = batch[3][:, :, 1]
        if mode == 'supervised':
            labels = all_labels
        else:
            labels = label_blocks[b_start:b_start+len(x_seq)]
            b_start += len(x_seq)
        labels = tf.where(tf.math.is_nan(labels), tf.zeros_like(labels), labels)
        if mode == 'glr':
            global_sample_len = int(x_seq.shape[1] * 0.3)
            rnd_t_g = np.random.randint(0, x_seq.shape[1] - global_sample_len)
            z_g = glr_model.global_encoder(x_seq[:, rnd_t_g:rnd_t_g + global_sample_len, :],
                                      mask=mask_seq[:, rnd_t_g:rnd_t_g + global_sample_len, :]).mean()
            pz_t = glr_model.local_encoder(x_seq, mask=mask_seq, window_size=window_size)
            z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
            lens = x_lens//glr_model.window_size
        elif mode == 'gpvae':
            z_t, _ = glr_model.encode(x_seq, mask_seq)
            z_g = None
            lens = x_lens//glr_model.window_size
        elif mode == 'vae':
            z_t = glr_model.encoder(x_seq, mask_seq, glr_model.window_size).sample()
            z_g = None
            lens = x_lens//glr_model.window_size
        elif mode in ['cdsvae', 'ours', 'ours2', 'ours3', 'ours4']:
            _, _, _, z_g, _, _, _, z_t, _, _, _, _, _, _ = glr_model(x_seq, mask_seq)
            lens = x_lens//glr_model.window_size
        elif mode == 'supervised':
            z_t = x_seq
            z_g = None
            lens = x_lens

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


def block_labels(dataset):
    all_labels_blocks = tf.concat([b[3][:, :, 1] for b in dataset], 0)
    all_labels_blocks = tf.split(all_labels_blocks, num_or_size_splits=all_labels_blocks.shape[1] // window_size, axis=1)
    all_labels_blocks = tf.stack([tf.math.reduce_sum(block, axis=1) for block in all_labels_blocks], axis=-1)
    return all_labels_blocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    parser.add_argument('--rep_size', type=int, default=8, help="Size of the representation vectors")
    parser.add_argument('--neptune', default='True', choices=('True', 'False'), help='activate neptune tracking')
    parser.add_argument('--f_dim', type=int, default=12, help="Size of static vector")
    parser.add_argument('--z_dim', type=int, default=4, help="Size of dynamic vector")
    parser.add_argument('--weight_rec', type=float, default=15., help='weighting on general reconstruction')
    parser.add_argument('--weight_f', type=float, default=7, help='weighting on KL to prior, content vector')
    parser.add_argument('--weight_z', type=float, default=1, help='weighting on KL to prior, motion vector')
    parser.add_argument('--weight_mi', default=2.5, type=float, help='weighting on mutual information loss')
    parser.add_argument('--weight_f_aug', default=.1, type=float, help='weighting on static contrastive loss')
    parser.add_argument('--weight_z_aug', default=.1, type=float, help='weighting on dynamic contrastive loss')
    parser.add_argument('--ce_epoch', default=150, type=int, help='warmup epochs for contrastive estimation')
    parser.add_argument('--note', default='sample', type=str, help='appx note')
    parser.add_argument('--mode', type=str, default='ours3')
    parser.add_argument('--seed', default=1948, type=int, help='random seed')
    args = parser.parse_args()

    # seeds = [1, 7, 13, 42, 1727, 1789, 1917, 1948, 2000, 2020, 2023]
    # for seed in seeds:
    #     print(f'seed: {seed}')
    # np.random.seed(args.seed)
    # tf.random.set_seed(args.seed)
    for _ in range(10):
        main(args)
