from model import LinearSeqModel, TransformerEncoderModel
import os
import torch


def test_forward():

	batch = 2
	seq_len = 10
	num_features = 6

	# Transformer forward-pass
	t = TransformerEncoderModel(seq_len=seq_len, num_features=num_features, d_model=32, nhead=4, num_layers=2, output_dim=6)
	t.eval()
	x = torch.randn(batch, seq_len, num_features)
	out_t = t(x)
	print("Transformer output shape:", out_t.shape)
	assert out_t.shape == (batch, 6)

	# LinearSeqModel forward-pass
	l = LinearSeqModel(input_seq_length=seq_len, latent_dim=12, output_dim=6)
	l.eval()
	out_l = l(x)
	print("LinearSeqModel output shape:", out_l.shape)
	assert out_l.shape == (batch, 6)


if __name__ == "__main__":
	test_forward()
	print("All forward tests passed")