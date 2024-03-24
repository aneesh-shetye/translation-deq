
import torch 
import torch.nn as nn 

def decoder(dec_layer: nn.TransformerDecoderLayer, 
			num_layers: int,
			norm = None):  
	
	return nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=num_layers, 
								 norm=norm)
	
def generate_square_subsequent_mask(size: int):
	"""
	output when size = 5
	tensor([[0., -inf, -inf, -inf, -inf],
		[0., 0., -inf, -inf, -inf],
		[0., 0., 0., -inf, -inf],
		[0., 0., 0., 0., -inf],
		[0., 0., 0., 0., 0.]])
	"""
	mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask

