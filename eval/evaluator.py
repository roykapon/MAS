import torch
from torch import nn
from data_loaders.dataset_utils import get_dims
from model.mdm import InputProcess, OutputProcess, PositionalEncoding
from utils import dist_utils


class VAE(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, motion, cond=None):
        mu, sigma = self.encoder(motion, cond)
        latent = mu + torch.exp(sigma / 2) * torch.randn_like(mu)
        recon = self.decoder(latent, cond)
        return {"mu": mu, "sigma": sigma, "recon_motion": recon}


def get_seq_mask(lengths):
    lengths = lengths.view(-1, 1)  # [bs, 1]
    positions = torch.arange(lengths.max(), device=lengths.device).view(1, -1)  # [1, nframes+1]
    return positions >= lengths  # [nframes+1, bs]


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_arch()
        self.setup_device()

    def set_arch(self):
        self.njoints, self.nfeats = get_dims(self.args.dataset)
        self.input_dim = self.njoints * self.nfeats
        self.latent_dim, self.num_layers, self.num_heads, self.ff_size, self.dropout, self.activation = self.args.e_latent_dim, self.args.e_num_layers, self.args.e_num_heads, self.args.e_ff_size, self.args.e_dropout, self.args.e_activation

        self.input_process = InputProcess(self.input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        self.mu_query = nn.Parameter(torch.randn([self.latent_dim]))
        self.sigma_query = nn.Parameter(torch.randn([self.latent_dim]))

    def setup_device(self):
        dist_utils.setup_dist(self.args.device)
        self.device = dist_utils.dev()
        self.to(self.device)

    def forward(self, x, cond=None):
        bs, njoints, nfeats, nframes = x.shape  # [bs, njoints, nfeats, nframes]
        assert njoints == self.njoints and nfeats == self.nfeats
        x = self.input_process(x)  # [nframes, bs, d]
        x = torch.cat((self.mu_query.expand(x[[0]].shape), self.sigma_query.expand(x[[0]].shape), x), axis=0)
        x = self.sequence_pos_encoder(x)  # [nframes+2, bs, d]

        # create a bigger mask, to allow to attend to mu and sigma
        x = self.seqTransEncoder(x, src_key_padding_mask=get_seq_mask(cond["y"]["lengths"] + 2))
        mu = x[0]
        logvar = x[1]

        return mu, logvar


class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_arch()
        self.setup_device()

    def set_arch(self):
        self.njoints, self.nfeats = get_dims(self.args.dataset)
        self.input_dim = self.njoints * self.nfeats
        self.latent_dim, self.num_layers, self.num_heads, self.ff_size, self.dropout, self.activation = self.args.e_latent_dim, self.args.e_num_layers, self.args.e_num_heads, self.args.e_ff_size, self.args.e_dropout, self.args.e_activation

        self.output_process = OutputProcess(self.input_dim, self.latent_dim, self.njoints, self.nfeats)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def setup_device(self):
        dist_utils.setup_dist(self.args.device)
        self.device = dist_utils.dev()
        self.to(self.device)

    def forward(self, latent, cond):
        bs, latent_dim = latent.shape  # [bs, d]
        nframes = cond["y"]["lengths"].max()
        assert latent_dim == self.latent_dim
        x = torch.zeros([nframes, bs, self.latent_dim], device=self.device)  # [nframes, bs, d]
        x = self.sequence_pos_encoder(x)  # [nframes, bs, d]

        x = self.seqTransDecoder(tgt=x, memory=latent.unsqueeze(0), tgt_key_padding_mask=get_seq_mask(cond["y"]["lengths"]))
        x = self.output_process(x)  # [nframes, bs, nfeats*njoints]

        return x


def create_evaluator(args):
    return VAE(args, TransformerEncoder(args), TransformerDecoder(args))
