# Transformer from Karpathy's minGPT (https://github.com/karpathy/minGPT)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here
    """

    def __init__(self, n_embd, n_head, pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        if isinstance(x, tuple):
            x = x[0]      
        #============================================================================
        # Batch size, sequence length, embedding dim
        B, T_batch, C = x.size()
        nh = self.n_head
        hs = C // nh

        # Create key, query, value tensors
        k = self.key(x).view(B, T_batch, nh, hs).transpose(1, 2)
        q = self.query(x).view(B, T_batch, nh, hs).transpose(1, 2)
        v = self.value(x).view(B, T_batch, nh, hs).transpose(1, 2)

        dropout_p = pdrop if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True).transpose(1, 2).contiguous().view(B, T_batch, C)

        y = self.resid_drop(self.proj(y))
        return y
        
class Block(nn.Module):
    """ a Transformer block """

    def __init__(self, n_embd, n_head, pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 6 * n_embd),
            nn.GELU(),
            nn.Linear(6 * n_embd, n_embd),
            nn.Dropout(pdrop),
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        att = self.attn(self.ln1(x))
        x = x + att
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, n_recur, n_layer, pdrop=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_in, n_embd)
        self.pos_emb = nn.Embedding(number_pos, n_embd)
        self.drop = nn.Dropout(pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, pdrop) for _ in range(n_layer)])
        # decoder head
        self.head = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab_out, bias=False))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self):
        """
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                    

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        return optim_groups

    def forward(self, idx, targets=None):
        """
        Returns:
            the logits in the final prediction; (batch_size, max_len, 2000)
        """        
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb(pos)
        x = self.drop(token_embeddings + position_embeddings)
                
        # collect the predicted logits (we add the option to use Recurrent/Looped Transformer)
        logits = []
        for reci in range(n_recur):
            for block in self.blocks:
                x = block(x) # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
                logits.append(self.head(x))

        return logits[-1]
    
    
# Determine the best available device
if torch.cuda.is_available():
    device = torch.device('cuda')          # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device('mps')           # Apple Silicon GPU (M1/M2/M3)
else:
    device = torch.device('cpu')           # Fallback to CPU

print(f"Using device: {device}")

# model config
#========================
number_pos = 1000  # maximum length of sequence our Transformer will be able to handle
vocab_in = 1001
vocab_out = 1001
n_embd = 576 
n_head = 8
n_recur = 1        # in case we want to use Recurrent(Looped) Transformer
n_layer = 8
pdrop = 0.1
#========================

model = GPT(n_embd, n_head, n_recur, n_layer, pdrop).to(device)
# model.load_state_dict(torch.load("logs/best_model.pt", map_location=device,weights_only=True))

model.train()
optim_groups = model.configure_optimizers()