import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from ReasoningCombinatorials.dfs_approach.sat.transformer_model import *

#========================================
total_steps = 3000000  # total training steps
lr = 1*1e-4            # initial Learning Rate
#========================================

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Optimizer
opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

# Scheduler
warmup_steps = 5
scheduler = get_linear_schedule_with_warmup(
    opt, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# optionally start from saved checkpoints
# opt.load_state_dict(torch.load("logs_sat/best_opt.pt",map_location=device))
# scheduler.load_state_dict(torch.load("logs_sat/best_scheduler.pt",map_location=device))