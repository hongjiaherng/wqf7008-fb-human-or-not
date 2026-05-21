from pytorch_tabnet.tab_network import TabNet

import torch
import torch.nn as nn


class TabNetWrapper(nn.Module):
    def __init__(self, input_dim, output_dim=1, device=None, n_d=8, n_a=8, n_steps=3, gamma=1.3, 
                 cat_idxs=[], cat_dims=[], cat_emb_dim=[]):
        super().__init__()
        self.tabnet = TabNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,                    # Width of the decision prediction layer
            n_a=n_a,                    # Width of the attention embedding layer
            n_steps=n_steps,            # Number of steps in the architecture
            gamma=gamma,                # Coefficient for feature reuse
            cat_idxs=cat_idxs,          # List of categorical features indices
            cat_dims=cat_dims,          # List of categorical features dimensions
            cat_emb_dim=cat_emb_dim,    # List of embeddings dimensions
            mask_type="sparsemax",      # "sparsemax" or "entmax"
            virtual_batch_size=32,
            group_attention_matrix=torch.eye(input_dim).to(device)
        )

    def forward(self, x):
        # Ensure input data is a FloatTensor (TabNet is strict about types)
        x = x.float() 
        
        # TabNet returns (output, M_loss). We drop M_loss (masking loss) 
        # so it safely integrates into the `train()` function.
        output, _ = self.tabnet(x)
        
        return output.squeeze(-1)