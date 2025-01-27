"""
Copyright (c) 2024, Weixin Liang.

This module demonstrates the initialization and usage patterns of the model blocks.
It serves as a reference implementation and is not intended for direct execution.
The examples below illustrate the proper way to instantiate the blocks and perform forward passes.
"""


from typing import Optional, Tuple, List, Callable, Dict, Any, Union

from src.model.BaselineDenseMamba import BaselineDenseMamba
from src.model.MixtureOfMamba import MixtureOfMamba



class MyMambaBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()

        self.dim = args.dim
        self.layer_id = layer_id

        self.logs: Optional[Dict] = None


        # For example, you need to use your own version of FusedRMSNorm. 
        self.norm = FusedRMSNorm(self.dim, eps=args.norm_eps, elementwise_affine=args.norm_affine)


        if args.use_MixtureOfMamba:
            # This is the Mixture-of-Mamba
            self.mamba_model = MixtureOfMamba(
                args, # args: ModelArgs,
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=self.dim, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
            assert args.use_BaselineDenseMamba is False, "args.use_BaselineDenseMamba and args.use_MixtureOfMamba cannot be True at the same time."

        elif args.use_BaselineDenseMamba:
            # This is the Baseline, Dense Mamba
            self.mamba_model = BaselineDenseMamba(
                args, # args: ModelArgs,
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=self.dim, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
            assert args.use_MixtureOfMamba is False, "args.use_BaselineDenseMamba and args.use_MixtureOfMamba cannot be True at the same time."

        else:
            raise NotImplementedError("should use either use_MixtureOfMamba or use_BaselineDenseMamba.")


        return 

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        image_data_indexes: Optional[torch.Tensor] = None,
        flex_block_mask = None,
        image_patch_masks = None,
    ):  
        

        batch_size, seq_len, dim = x.shape 

        modality_indices = compute_modality_indices(image_patch_masks)
        modality_masks = modality_indices 



        hidden_states = x 
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mamba_model(
            hidden_states, 
            modality_masks=modality_masks, 

        )
        hidden_states = residual + hidden_states
        out = hidden_states
        
        return out, None





class BaselineDenseMambaStack(torch.nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()

        # Initialize two MyMambaBlock instances
        self.block1 = MyMambaBlock(args, layer_id=layer_id)
        self.block2 = MyMambaBlock(args, layer_id=layer_id)


        self.dim = args.dim
        self.layer_id = layer_id
        self.logs: Optional[Dict] = None


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        image_data_indexes: Optional[torch.Tensor] = None,
        flex_block_mask = None,
        image_patch_masks = None,
    ):
        # Pass input through the first block
        out, _ = self.block1(
            x,
            mask=mask,
            freqs_cis=freqs_cis,
            cache=cache,
            image_data_indexes=image_data_indexes,
            flex_block_mask=flex_block_mask,
            image_patch_masks=image_patch_masks,
        )

        # Pass the output of the first block through the second block
        out, _ = self.block2(
            out,
            mask=mask,
            freqs_cis=freqs_cis,
            cache=cache,
            image_data_indexes=image_data_indexes,
            flex_block_mask=flex_block_mask,
            image_patch_masks=image_patch_masks,
        )

        # Return the final output
        return out, None

