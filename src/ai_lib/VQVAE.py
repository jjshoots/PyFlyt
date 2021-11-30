#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

try:
    from .neural_blocks import *
except:
    from neural_blocks.neural_blocks import *



class VQVAE(nn.Module):
    """
    VQVAE model
    Basically the state encoder
    """
    def __init__(self, num_embedding, embedding_dim, independent_codebook=False):
        super().__init__()

        self.num_latents = 16
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim)
        self.quantizer = VQ_EMA(
            independent_codebook=independent_codebook,
            embedding_dim=embedding_dim,
            num_embeddings=num_embedding,
            num_latents=self.num_latents
            )


    def encode(self, input):
        return self.encoder(input)


    def decode(self, input):
        return self.decoder(input)


    def quantize(self, input):
        return self.quantizer(input)


    def encoding_indices_to_encoding(self, encoding_indices):
        shape = encoding_indices.shape
        encoding = torch.zeros(*shape[:2], self.num_embedding, *shape[-2:]).to(encoding_indices.device)
        encoding = encoding.scatter(2, encoding_indices.unsqueeze(2), 1.)
        return encoding


    def encoding_to_quantized(self, encoding):
        encoding_indices = torch.argmax(encoding, dim=2)
        return self.encoding_indices_to_quantized(encoding_indices)


    def encoding_indices_to_quantized(self, encoding_indices):
        shape = encoding_indices.shape
        encoding_indices = encoding_indices.view(encoding_indices.shape[0] * encoding_indices.shape[1], -1)
        quantized = self.quantizer.decode(encoding_indices)

        return quantized.view(*shape[:2], -1, *shape[-2:])



class Encoder(nn.Module):
    """
    Encoder for VQVAE
    """
    def __init__(self, embedding_dim):
        super().__init__()

        _channels_description = [3, 64, 64, 128, 128, embedding_dim]
        _kernels_description = [3, 3, 3, 3, 1]
        _pooling_description = [2, 2, 2, 2, 0]
        _activation_description = ['lrelu'] * (len(_kernels_description) - 1) + ['sigmoid']
        self.model = Neural_blocks.generate_conv_stack(_channels_description, _kernels_description, _pooling_description, _activation_description)


    def forward(self, input):
        seq_bat = input.shape[:2]
        input = input.flatten(0, 1)

        output = self.model(input)
        output = output.view(*seq_bat, *output.shape[1:])
        return output



class Decoder(nn.Module):
    """
    Decoder for VQVAE
    """
    def __init__(self, embedding_dim):
        super().__init__()

        _channels = [embedding_dim, 512, 256, 128, 64]
        _kernels = [4, 4, 4, 4]
        _padding = [1, 1, 1, 1]
        _stride = [2, 2, 2, 2]
        _activation = ['lrelu', 'lrelu', 'lrelu', 'lrelu']
        self.unsqueeze = Neural_blocks.generate_deconv_stack(_channels, _kernels, _padding, _stride, _activation)

        # the final image, 3x64x64
        _channels = [64, 32, 3]
        _kernels = [3, 1]
        _pooling = [0, 0]
        _activation = ['lrelu', 'sigmoid']
        self.regenerate = Neural_blocks.generate_conv_stack(_channels, _kernels, _pooling, _activation)


    def forward(self, input):
        input_shape = input.shape
        input = input.flatten(0, 1)

        output = self.unsqueeze(input)
        output = self.regenerate(output)

        output = output.view(*input_shape[:2], *output.shape[1:])
        return output


class VQ_EMA(nn.Module):
    def __init__(self, independent_codebook, embedding_dim, num_embeddings, num_latents, decay=0.99, epsilon=1e-6):
        super().__init__()

        self._num_latents = num_latents
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._decay = decay
        self._epsilon = epsilon
        self._independent_codebook = independent_codebook

        latent_offset = torch.tensor(0)
        if independent_codebook:
            latent_offset = torch.arange(0, self._num_latents*self._num_embeddings, self._num_embeddings)
        else:
            self._num_latents = 1

        # initialize all codebooks identically
        embedding = 0.01 * torch.randn(self._embedding_dim, self._num_embeddings)
        embedding = torch.stack([embedding] * self._num_latents, dim=-1)

        self.register_buffer('_latent_offset', latent_offset, persistent=False)
        self.register_buffer('_embedding', embedding)
        self.register_buffer('_dw', torch.zeros(self._embedding_dim, self._num_embeddings, self._num_latents))
        self.register_buffer('_cluster_size', torch.zeros(self._num_embeddings, self._num_latents))

        # do this so VIM coc plays nicely with syntax
        self._latent_offset = self._latent_offset
        self._embedding = self._embedding
        self._dw = self._dw
        self._cluster_size = self._cluster_size


    def forward(self, input):
        # squeeze batch and sequence dimension together
        # squeeze HW together
        input_shape = input.shape
        input = input.view(input.shape[0] * input.shape[1], -1, input.shape[3] * input.shape[4])

        # calculate distances
        distances = self._embedding.unsqueeze(0) - input.unsqueeze(-2)
        distances = torch.linalg.norm(distances, dim=1)

        # get encoding indices
        encoding_indices = torch.argmin(distances, dim=1)

        # get the encoding from the indices
        encoding = torch.zeros(distances.shape, device=input.device)
        encoding = encoding.scatter(1, encoding_indices.unsqueeze(1), 1)

        # quantize
        quantized = self.decode(encoding_indices)

        if self.training:
            # Laplce smoothing to handle division by 0
            n = torch.sum(self._cluster_size, -1).unsqueeze(-1) + 1e-6
            self._cluster_size = (self._cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n

            dw = None
            if self._independent_codebook:
                # update cluster size using ema
                self._cluster_size = self.ema(self._cluster_size, torch.sum(encoding, dim=0))
                # get innovation
                dw = torch.sum(encoding.unsqueeze(1) * input.unsqueeze(2), dim=0)

                # perplexity, divide by self._num_embeddings to make it num_embeddings agnostic
                # theoretical limit for this value is 1.0 for uniformly distributed codebook usage
                avg_probs = torch.mean(encoding, 0)
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=0)).detach() / self._num_embeddings

            else:
                # update cluster size using ema
                self._cluster_size = self.ema(self._cluster_size, torch.sum(torch.sum(encoding, dim=0), dim=-1, keepdim=True))
                # get innovation
                dw = torch.sum(torch.sum(encoding.unsqueeze(1) * input.unsqueeze(2), dim=0), dim=-1, keepdim=True)

                # perplexity, divide by self._num_embeddings to make it num_embeddings agnostic
                # theoretical limit for this value is 1.0 for uniformly distributed codebook usage
                avg_probs = torch.mean(torch.mean(encoding, 0), -1)
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=0, keepdim=True)).detach() / self._num_embeddings


            # update innovation using ema
            self._dw = self.ema(self._dw, dw / self._cluster_size.unsqueeze(0)).data

            # update embedding using innovation and ema
            self._embedding = self.ema(self._embedding, self._dw)

            # commitment loss
            commitment_loss = F.mse_loss(quantized.data, input)

            # straight through estimator
            quantized = input + (quantized - input).data

            # reshape quantized to input shape
            quantized = quantized.view(input_shape)
            encoding = encoding.view(*input_shape[:2], -1, *input_shape[-2:])

            return quantized, encoding, commitment_loss, perplexity
        else:
            # reshape quantized to input shape
            quantized = quantized.view(input_shape)
            encoding = encoding.view(*input_shape[:2], -1, *input_shape[-2:])
            return quantized, encoding, None, None


    def ema(self, value, update):
        return self._decay * value + (1 - self._decay) * update


    def decode(self, encoding_indices):
        """
        use offsets to select embeddings, credit https://github.com/andreaskoepf
        """
        shape = encoding_indices.shape

        # add offsets to indices
        encoding_indices = encoding_indices + self._latent_offset.unsqueeze(0)
        encoding_indices = encoding_indices.view(-1)

        # quantized
        quantized = self._embedding.permute(2, 1, 0).reshape(-1, self._embedding_dim)[encoding_indices]
        quantized = quantized.view(*shape, self._embedding_dim).permute(0, 2, 1).contiguous()

        return quantized
