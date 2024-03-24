import torch 
import torch.nn as nn 

def encoder(enc_layer: nn.TransformerEncoderLayer, 
            num_layers: int,
            norm = None, 
            mask_check: bool = True): 
    
    return nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=num_layers, 
                                 norm=norm, mask_check=mask_check)