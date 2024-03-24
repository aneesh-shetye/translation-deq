# import random 

# import torch 
# import torch.nn as nn 

# from dataloader_utils import MyCollate
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from transformers import BertModel
# from datasets import load_dataset
# from decoder import decoder
# from model import Model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# #setting seed 
# MANUAL_SEED = 3407
# random.seed(MANUAL_SEED)
# torch.manual_seed(MANUAL_SEED)
# torch.backends.cudnn.deterministic = True


# # ARGS: 

# ## TRAIN-ARGS: 
# ### batch_size = 32
# ### epochs = 10
# ### tol = 1e-3

# ## DATASET SPECS:  
# ### dataset_name = "bentrevett/multi30k"

# ## MODEL ARGS: 

 
# #initializing dataset
# dataset = load_dataset("bentrevett/multi30k")

# #initializing tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# print(tokenizer.vocab_size)

# #initializing dataloader 
# ##train_loader
# train_loader = DataLoader(dataset=dataset['train'], batch_size=16, collate_fn=MyCollate(tokenizer)) 
# val_loader = DataLoader(dataset=dataset['validation'], batch_size=16, collate_fn=MyCollate(tokenizer)) 

# #defining encoder 
# enc = BertModel.from_pretrained("bert-base-multilingual-cased")

# #defining decoder
# dec_layer = nn.TransformerDecoderLayer(768,nhead=4)
# dec = decoder(dec_layer=dec_layer,num_layers=3)

# # model = Model(enc,dec)
# pad_idx=0
# print(len(train_loader))
# for batch in enumerate(train_loader):
#     # print(batch)
#     src = batch[1][0]
#     tgt = batch[1][1]
#     print(src)
#     #ensuring src is of the right size: 
#     if len(src.shape)>2: 
#         src = src.squeeze(-1)
    

#     #defining att_mask
#     att_mask = torch.ones(src.shape).masked_fill(src == pad_idx,0)
#     att_mask = att_mask.squeeze(-1) #att_mask.shape = (batch_size, src_len)

#     att_mask2 = torch.ones(tgt.shape).masked_fill(tgt == pad_idx,0)
#     att_mask2 = att_mask2.squeeze(-1)

#     print(f'src.shape = {src.shape}, att_mask.shape = {att_mask.shape}')
    
#     #getting src embeddings 
#     mem = enc(src, attention_mask = att_mask)['last_hidden_state'] 
#     # print(mem)
#     print(f'mem.shape= {mem.shape}')

#     #getting tgt embeddings 
#     tgt = enc(tgt, attention_mask = att_mask2)['last_hidden_state'] 
#     print("TGT",tgt.shape)
#     # print(tgt)
#     tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])

#     pred = dec(tgt=tgt, memory=mem, memory_mask=att_mask, tgt_mask=tgt_mask,
#                tgt_is_causal=True, memory_is_causal=True)
    
#     print(f'pred.shape= {pred.shape}')

#     break


# import jax.numpy as jnp
# def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
#   x0 = z_init
#   x1 = f(x0)
#   x2 = f(x1)
#   X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
#   F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

#   res = []
#   for k in range(2, max_iter):
#     n = min(k, m)
#     G = F[:n] - X[:n]
#     GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
#     print(G.shape)
#     print(GTG.shape)
#     H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
#                    [ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
#     print(H.shape)
#     alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]
#     print(alpha.shape)
#     print(F[:n].shape)
#     xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n])
#     X = X.at[k % m].set(xk)
#     F = F.at[k % m].set(f(xk))

#     res = jnp.linalg.norm(F[k % m] - X[k % m]) / (1e-5 + jnp.linalg.norm(F[k % m]))
#     if res < tol:
#       break
#   return xk

# def fixed_point_layer(solver, f, params, x):
#   z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
#   return z_star

# f = lambda W, x, z: jnp.tanh(jnp.dot(W, z) + x)

# from jax import random

# ndim = 10
# W = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
# x = random.normal(random.PRNGKey(1), (ndim,))
# # print(x.shape)
# z_star = fixed_point_layer(anderson_solver, f, W, x)
# print(z_star)
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))
    
def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    print(H.shape)
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        print(y[:,:n+1].shape)
        # alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]  
        alpha = torch.linalg.solve(H[:,:n+1,:n+1],y[:,:n+1])[:,1:n+1,0]   # (bsz x n)
 # (bsz x n)
        print("ALPHA",alpha.shape)
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res

import torch.autograd as autograd

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z
    
f = ResNetLayer(64,128)
deq = DEQFixedPoint(f, anderson, tol=1e-4, max_iter=100, beta=2.0)
X = torch.randn(10,64,32,32)
out = deq(X)