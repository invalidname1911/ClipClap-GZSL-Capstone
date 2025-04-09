# system, numpy
import os
import sys
import numpy as np
import math
from einops import rearrange, repeat
import einops
import opt_einsum
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# user defined
from src.optimizer import SAM

torch.set_printoptions(threshold=10_000)
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for audio-visual feature fusion
    
    Parameters:
    -----------
    dim : int
        Input dimension of the feature
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability for attention weights
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Forward pass for multi-head self-attention
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, dim)
            For our case, seq_len=1 as we're working with feature vectors
            
        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as input after self-attention
        """
        # Apply layer normalization
        residual = x
        x = self.layer_norm(x)
        
        batch_size = x.size(0)
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        # Add residual connection
        output = output + residual
        
        # Remove the extra dimension (seq_len=1)
        output = output.squeeze(1)
        
        return output


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, hidden_size=-1):
        super(EmbeddingNet, self).__init__()
        modules = []

        if hidden_size > 0:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)













































class ClipClap_model(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(ClipClap_model, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        self.input_dim_audio = input_size_audio
        self.input_dim_video = input_size_video

        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.drop_proj_o=params_model['dropout_decoder']
        self.drop_proj_w=params_model['additional_dropout']
        self.reg_loss=params_model['reg_loss']
        self.cross_entropy_loss=params_model['cross_entropy_loss']
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.drop_enc=params_model['dropout_encoder']
        
        # Multi-head attention parameters
        self.use_mhsa = params_model.get('use_mhsa', False)  # Whether to use MHSA (defaults to False)
        self.mhsa_num_heads = params_model.get('mhsa_num_heads', 8)  # Number of attention heads
        self.mhsa_dropout = params_model.get('mhsa_dropout', 0.1)  # Dropout for attention

        self.rec_loss = params_model['rec_loss']

        self.lr_scheduler = params_model['lr_scheduler']

        print('Initializing trainable models...', end='')


        self.modality = params_model['modality']
        self.word_embeddings = params_model['word_embeddings']

        if self.modality == 'audio':
            self.O_enc = EmbeddingNet(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            self.W_enc = EmbeddingNet(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
        elif self.modality == 'video':
            self.O_enc = EmbeddingNet(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            self.W_enc = EmbeddingNet(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
        else:  # audio-visual case
            # For audio-visual mode, we may add a Multi-Head Self-Attention layer
            self.input_dim_av = 1536  # Combined dimension of audio (1024) and visual (512) features
            
            if self.use_mhsa:
                # Initialize MHSA layer for audio-visual features
                self.mhsa = MultiHeadSelfAttention(
                    dim=self.input_dim_av,
                    num_heads=self.mhsa_num_heads,
                    dropout=self.mhsa_dropout
                )
                print(f' Added MHSA with {self.mhsa_num_heads} heads...', end='')
            
            self.O_enc = EmbeddingNet(
                input_size=1536,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            w_in_dim = 1536
            if self.word_embeddings == 'wavcaps':
                w_in_dim = 1024
            elif self.word_embeddings == 'clip':
                w_in_dim = 512

            self.W_enc = EmbeddingNet(
                input_size=w_in_dim,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )




        word_embedding_dim = 512
        self.O_proj = EmbeddingNet(
            input_size=512,
            hidden_size=self.hidden_size_decoder,
            output_size=self.dim_out,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )
        self.D_o = EmbeddingNet(
            input_size=self.dim_out,
            hidden_size=self.hidden_size_decoder,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )


        self.W_proj= EmbeddingNet(
            input_size=word_embedding_dim,
            output_size=self.dim_out,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )

        self.D_w = EmbeddingNet(
            input_size=self.dim_out,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )









        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']

        optimizer = params_model['optimizer']
        self.is_sam_optim = False
        if optimizer == 'adam':
            self.optimizer_gen = optim.Adam(
                self.parameters(),
                lr=self.lr, weight_decay=1e-5
            )
            if self.lr_scheduler:
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        elif optimizer == 'adam-sam':
            self.optimizer_gen = SAM(self.parameters(), optim.Adam, lr=self.lr, weight_decay=1e-5)
            self.is_sam_optim = True
            if self.lr_scheduler:
                # lr scheduling on base optimizer
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen.base_optimizer, 'max', patience=3, verbose=True)
        else:
            raise NotImplementedError

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_cyc = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        print('Done')

    def optimize_scheduler(self, value):
        if self.lr_scheduler:
            self.scheduler_learning_rate.step(value)

    def forward(self, a, v, w, masks, timesteps):
        b, _ = a.shape
        device = a.device
        v = v.type(torch.float32)
        if self.modality == 'audio':
            w = w[:,512:]
            model_input = a

        elif self.modality == 'video':
            w = w[:,:512]
            model_input = v
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input = torch.cat((v, a), dim=1)
            
            # Apply Multi-Head Self-Attention if enabled
            if self.use_mhsa:
                # MHSA expects input shape [batch_size, seq_len, dim]
                # For feature vectors, we add a dummy sequence dimension
                model_input = model_input.unsqueeze(1)
                model_input = self.mhsa(model_input)


        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)


        rho_o = self.D_o(theta_o)


        theta_w = self.W_proj(w)


        rho_w=self.D_w(theta_w)


        output = {
            "theta_w": theta_w,
            "w": w,
            "rho_w": rho_w,
            "theta_o": theta_o,
            "rho_o": rho_o,
        }


        return output


    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy):

        theta_w = outputs['theta_w']

        w = outputs['w']
        rho_w = outputs['rho_w']

        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']


        device = theta_w.device

        if self.cross_entropy_loss==True:
            if self.modality == 'audio':
                embeddings_crossentropy = embeddings_crossentropy[:,512:]
            elif self.modality == 'video':
                embeddings_crossentropy = embeddings_crossentropy[:,:512]
            else:
                if self.word_embeddings == 'wavcaps':
                    embeddings_crossentropy = embeddings_crossentropy[:,512:]
                elif self.word_embeddings == 'clip':
                    embeddings_crossentropy = embeddings_crossentropy[:,:512]

            embedding_cross_entropy=self.W_proj(self.W_enc(embeddings_crossentropy))
            Cross_loss=nn.CrossEntropyLoss()
            scores=torch.matmul(theta_o, embedding_cross_entropy.t()) # (bs, 64) x (K_seen, 64).T = (bs, 64) x (64, K_seen) = (bs, K_seen)
            # gt_cross_entropy = [1, 3, 2, 55, 97, 45, ...] list of gt class labels -> shape (bs,)
            l_ce=Cross_loss(scores, gt_cross_entropy)
        else:
            l_ce = torch.tensor(0., device=device)

        if self.reg_loss==True:
            l_reg = (
                self.MSE_loss(theta_o, theta_w)
            )
        else:
            l_reg = torch.tensor(0., device=device)


        if self.rec_loss == True:
            l_rec = (
                    self.MSE_loss(w, rho_o) +
                    self.MSE_loss(w, rho_w)
            )
        else:
            l_rec = torch.tensor(0., device=device)


        loss_total = l_rec+l_reg+l_ce
        loss_dict = {
            "Loss/total_loss": loss_total.detach().cpu(),
            "Loss/loss_reg": l_reg.detach().cpu(),
            "Loss/loss_cmd_rec": l_rec.detach().cpu(),
            "Loss/cross_entropy": l_ce.detach().cpu()

        }
        return loss_total, loss_dict

    # cls_numeric = class index
    # cls_embedding = w2v embedding of the target
    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=False):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)

            # Backward pass
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(audio, video, cls_embedding, masks, timesteps)
                second_loss, _ = self.compute_loss(outputs_second, embedding_crossentropy,  cls_numeric)
                second_loss.backward()
                self.optimizer_gen.second_step(zero_grad=True)

        return loss_numeric, loss

    def get_embeddings(self, a, v, w, masks, timesteps):
        b, _ = a.shape
        device = a.device
        v = v.type(torch.float32)



        if self.modality == 'audio':
            w = w[:,512:]
            model_input = a

        elif self.modality == 'video':
            w = w[:,:512]
            model_input = v
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input = torch.cat((v, a), dim=1)


        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)

        theta_w=self.W_proj(w)

        return theta_o, theta_o, theta_w
