import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from mutual_info import log_density, logsumexp

class Encoder(nn.Module):
    def __init__(self, hidden_sizes):
        """Initializes the instance"""
        super(Encoder, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.fc = nn.Sequential(*[nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i+1], bias=True) for i in range(len(hidden_sizes)-1)])

    def forward(self, x):
        # x is in batch x time x feature
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, output_size, output_length, hidden_sizes):
        """Initializes the instance"""
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.output_length = output_length
        self.hidden_sizes = hidden_sizes

        self.embedding = nn.Sequential(nn.Linear(hidden_sizes[0], hidden_sizes[0]), nn.Tanh())
        self.fc = nn.Sequential(*[nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(1, len(hidden_sizes)-1)])
        self.rnn = nn.LSTM(hidden_sizes[0], hidden_sizes[0], batch_first=True)
        self.mean_gen = nn.Linear(hidden_sizes[-1], self.output_size)
        self.cov_gen = nn.Sequential(nn.Linear(hidden_sizes[-1], self.output_size), nn.Sigmoid())

    def forward(self, z_t):
        """Reconstruct a time series from the representations of all windows over time

        Args:
            z_t: Representation of signal windows with shape [batch_size, n_windows, representation_size]
        """
        n_batch, prior_len, _ = z_t.size()
        emb = self.embedding(z_t)
        recon_seq = []
        for t in range(z_t.size(1)):
            rnn_out, _ = self.rnn(torch.randn(len(z_t), self.output_length, self.hidden_sizes[0]), (emb[:, t, :].unsqueeze(0), emb[:, t, :].unsqueeze(0)))
            recon_seq.append(rnn_out)
        recon_seq = torch.cat(recon_seq, 1)
        recon_seq = self.fc(recon_seq)
        x_mean = self.mean_gen(recon_seq)
        x_cov = self.cov_gen(recon_seq)
        return dist.Normal(loc=x_mean, scale=x_cov*0.5)

class CDSVAE(nn.Module):
    def __init__(self, encoder, decoder, configs, args):
        """Initializes the instance"""
        super(CDSVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.feature_dim = configs["feature_size"]              # dim of features (10 physionet, 10 air_quailty)
        self.sample_len = configs["t_len"]                      # dim of time (60 physionet, 672 air_quality)
        self.window_size = configs["window_size"]               # split time series to windows of this size
        self.dataset_size = 1

        self.fc_dim = encoder.hidden_sizes[-1]                  # dim of FC encoder/decoder output/input
        self.hidden_dim = 32                       # dim of latent information
        self.f_dim = args.f_dim                                 # dim of latent static
        self.z_dim = args.z_dim                                 # dim of latent dynamic

        self.M = configs["mc_samples"]                          # used for tiling information ??
        self.weight_rec = args.weight_rec
        self.weight_f = args.weight_f
        self.weight_z = args.weight_z
        self.weight_mi = args.weight_mi

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # ----- Posterior of static and dynamic
        # static and dynamic features share one lstm
        self.fz_lstm = tf.keras.layers.LSTM(units=self.fc_dim, activation=tf.nn.sigmoid)
        self.f_mean = tf.keras.layers.Dense(self.f_dim, dtype=tf.float32)
        self.f_logvar = tf.keras.layers.Dense(self.f_dim, dtype=tf.float32)

        # dynamic features from the next lstm
        self.z_lstm = tf.keras.layers.LSTM(units=self.fc_dim, activation=tf.nn.sigmoid, return_sequences=True)
        self.z_mean = tf.keras.layers.Dense(self.z_dim, dtype=tf.float32)
        self.z_logvar = tf.keras.layers.Dense(self.z_dim, dtype=tf.float32)

    def __call__(self, x_seq, mask_seq=None):

        # print(f'FWD: x_seq={x_seq.shape}, mask_seq={mask_seq.shape}')

        # encode and sample post
        f_mean, f_logvar, pf_post, z_mean_post, z_logvar_post, pz_post = \
            self.encode_and_sample_post(x_seq, mask_seq, self.window_size, random_sampling=True)
        f_post, z_post = pf_post.sample(), pz_post.sample()

        # print(f'FWD: f_mean={f_mean.shape}, f_logvar={f_logvar.shape}, f_post={f_post.shape},'
        #       f'z_mean={z_mean_post.shape}, z_logvar={z_logvar_post.shape}, z_post={z_post.shape}')

        # sample prior
        z_mean_prior, z_logvar_prior, pz_prior, z_prior = self.sample_motion_prior_train(z_post, random_sampling=True)
        # print(f'FWD: z_mean_prior={z_mean_prior.shape}, z_logvar_prior={z_logvar_prior.shape}, '
        #       f'z_prior={z_prior.shape}')

        f_expand = tf.repeat(tf.expand_dims(f_post, axis=1), repeats=z_post.shape[1], axis=1)
        # print(f'FWD: f_expand={f_expand.shape}')
        zf = tf.concat([z_post, f_expand], axis=2)
        # print(f'FWD: zf={zf.shape}')

        px_hat = self.decoder(zf)
        recon_x = px_hat.sample()
        # print(f'FWD: recon_x={recon_x.shape}')
        # input('press any key...')

        return f_mean, f_logvar, pf_post, f_post, z_mean_post, z_logvar_post, pz_post, z_post, \
               z_mean_prior, z_logvar_prior, pz_prior, z_prior, px_hat, recon_x

    def encode_and_sample_post(self, x, mask, window_size, random_sampling):
        # pipeline: x (fc_enc)-> fc_x (z_lstm, z_rnn)-> (f_post, z_post)
        # x is in batch x time x features

        # handle mask
        if not mask is None:
            mask = (tf.reduce_sum(mask, axis=-1) < int(0.7 * x.shape[-1]))

        # FC encoder
        fc_x = self.encoder(x)

        # LSTM encoder (static)
        lstm_out_f = []
        for t in range(0, x.shape[1] - window_size + 1, window_size):
            if not mask is None:
                x_mapped = self.fz_lstm(fc_x[:, t:t + window_size, :], mask=mask[:, t:t + window_size])
            else:
                x_mapped = self.fz_lstm(fc_x[:, t:t + window_size, :])
            lstm_out_f.append(x_mapped)
        lstm_out_f = tf.stack(lstm_out_f, axis=1)
        f_mean = self.f_mean(lstm_out_f[:, -1])
        f_logvar = tf.nn.softplus(self.f_logvar(lstm_out_f[:, -1]))
        pf_post = tfd.MultivariateNormalDiag(loc=f_mean, scale_diag=f_logvar) if random_sampling else f_mean

        # LSTM encoder (dynamic)
        lstm_out_z = self.z_lstm(lstm_out_f)
        z_mean = self.z_mean(lstm_out_z)
        z_logvar = self.z_logvar(lstm_out_z)
        pz_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=z_logvar) if random_sampling else z_mean

        return f_mean, f_logvar, pf_post, z_mean, z_logvar, pz_post

    # ------ sample z from learned LSTM prior base on previous postior, teacher forcing for training  ------
    def sample_motion_prior_train(self, z_post, random_sampling=True):
        # z_post is in batch x (new) time x features
        z_out = None
        z_means = None
        z_logvars = None
        batch_size, frames = z_post.shape[0], z_post.shape[1]

        z_t = tf.zeros((batch_size, self.z_dim))
        h_t_ly1 = tf.zeros((batch_size, self.hidden_dim))
        c_t_ly1 = tf.zeros((batch_size, self.hidden_dim))
        h_t_ly2 = tf.zeros((batch_size, self.hidden_dim))
        c_t_ly2 = tf.zeros((batch_size, self.hidden_dim))

        for i in range(frames):
            _, (h_t_ly1, c_t_ly1) = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            _, (h_t_ly2, c_t_ly2) = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = tfd.MultivariateNormalDiag(loc=z_mean_t, scale_diag=z_logvar_t).sample() \
                if random_sampling else z_mean_t

            if z_out is None:
                z_out = tf.expand_dims(z_prior, axis=1)
                z_means = tf.expand_dims(z_mean_t, axis=1)
                z_logvars = tf.expand_dims(z_logvar_t, axis=1)
            else:
                z_out = tf.concat((z_out, tf.expand_dims(z_prior, axis=1)), axis=1)
                z_means = tf.concat((z_means, tf.expand_dims(z_mean_t, axis=1)), axis=1)
                z_logvars = tf.concat((z_logvars, tf.expand_dims(z_logvar_t, axis=1)), axis=1)
            z_t = z_post[:, i, :]
        pz_out = tfd.MultivariateNormalDiag(loc=z_means, scale_diag=z_logvars) if random_sampling else z_means
        return z_means, z_logvars, pz_out, z_out

    def compute_loss(self, x, m_mask=None, return_parts=False, **kwargs):
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)
        x = tf.tile(x, [self.M, 1, 1])  # shape=(M*BS, TL, D)

        if m_mask is not None:
            m_mask = tf.identity(m_mask)
            m_mask = tf.cast(m_mask, dtype=tf.float32)
            m_mask = tf.tile(m_mask, [self.M, 1, 1])  # shape=(M*BS, TL, D)

        f_mean, f_logvar, pf_post, f_post, z_mean_post, z_logvar_post, pz_post, z_post, \
        z_mean_prior, z_logvar_prior, pz_prior, z_prior, px_hat, recon_x = self(x, m_mask)

        # compute losses

        # reconstruction loss
        nll = -px_hat.log_prob(x)
        if m_mask is not None:
            nll = tf.where(m_mask == 1, tf.zeros_like(nll), nll)

        # KL divergence of f and z_t
        pg = tfd.MultivariateNormalDiag(loc=tf.zeros((f_post.shape[-1])), scale_diag=tf.ones((f_post.shape[-1])))
        kld_f = tfd.kl_divergence(pf_post, pg)
        kld_z = tfd.kl_divergence(pz_post, pz_prior)

        # Mutual information I(f; z_t) via MWS
        n_batch, n_seq = z_post.shape[0:2]
        expand = lambda x : tf.repeat(tf.expand_dims(x, axis=0), repeats=n_seq, axis=0)

        # seq x batch x batch x f_feats
        _logq_f_tmp = log_density(tf.expand_dims(expand(f_post), axis=2),
                                  tf.expand_dims(expand(f_mean), axis=1),
                                  tf.expand_dims(expand(f_logvar), axis=1))

        # seq x batch x batch x z_feats
        _logq_z_tmp = log_density(tf.expand_dims(tf.transpose(z_post, perm=[1, 0, 2]), axis=2),
                                  tf.expand_dims(tf.transpose(z_mean_post, perm=[1, 0, 2]), axis=1),
                                  tf.expand_dims(tf.transpose(z_logvar_post, perm=[1, 0, 2]), axis=1))

        # seq x batch x batch x f_feats + z_feats
        _logq_fz_tmp = tf.concat((_logq_f_tmp, _logq_z_tmp), axis=3)

        # seq x batch, seq x batch, seq x batch
        const = tf.math.log(float(n_batch * self.dataset_size))
        logq_f = logsumexp(tf.reduce_sum(_logq_f_tmp, axis=3), axis=2, keepdims=False) - const
        logq_z = logsumexp(tf.reduce_sum(_logq_z_tmp, axis=3), axis=2, keepdims=False) - const
        logq_fz = logsumexp(tf.reduce_sum(_logq_fz_tmp, axis=3), axis=2, keepdims=False) - const

        mi_fz = tf.reduce_mean(tf.nn.relu(logq_fz - logq_f - logq_z))

        kld_f = tf.reduce_mean(kld_f, axis=[-1]) / (x.shape[-1])
        kld_z = tf.reduce_mean(kld_z, axis=[-1]) / (x.shape[-1])
        nll = tf.reduce_mean(nll, axis=[1, 2])

        elbo = - nll * self.weight_rec - kld_f * self.weight_f - kld_z * self.weight_z + mi_fz * self.weight_mi
        elbo = tf.reduce_mean(elbo)
        if return_parts:
            return -elbo, tf.reduce_mean(nll), tf.reduce_mean(kld_f), tf.reduce_mean(kld_z), tf.reduce_mean(mi_fz)
        return -elbo

    def get_trainable_vars(self):
        self.compute_loss(x=tf.random.normal(shape=(1, self.sample_len, self.feature_dim), dtype=tf.float32),
                          m_mask=tf.zeros(shape=(1, self.sample_len, self.feature_dim), dtype=tf.float32))
        return self.trainable_variables