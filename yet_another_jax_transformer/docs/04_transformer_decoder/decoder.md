# The Transformer Decoder

As for the encoder, the decoder is a stack of identical blocks. Again, then, let's first define the single decoder block. You can refer to the image (right) in Section 1 to see all the components we need to code to make the decoder work. 

The Decoder block is composed of:
1. a Masked Multi-Head Attention layer. Here, "masked" refers to the autoregressive property of self-attention in the decoder. Specifically, we want to force an arbitrary token at position *i* to express a non-zero attention weight to preceding tokens. But we don't have to take care of it now: the trick 💡 is to use a particular attention mask, which we will see later.
2. a Cross-Attention layer. Here is where the magic happens. The decoder receives the Queries and Keys from the encoder. We will call them "memory."
3. a Feed-Forward layer with element-wise non-linear activation.
4. Skip connections and Layer Normalization after each sub-layers.

## Decoder Block

```python
class DecoderBlock(hk.Module):
    """
    Transformer decoder block.

    :param d_model: dimension of the model.
    :param num_heads: number of attention heads.
    :param d_ff: dimension of the feedforward network model.
    :param p_dropout: dropout rate.
    """

    def __init__(self, d_model, num_heads, d_ff, p_dropout, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout

        # self-attention sub-layer
        self.self_attn = MultiheadAttention(
            d_model=self.d_model, num_heads=self.num_heads
        )
        # src-target cross-attention sub-layer
        self.cross_attn = MultiheadAttention(
            d_model=self.d_model, num_heads=self.num_heads
        )
        # positionwise feedforward sub-layer
        self.ff = PositionwiseFeedForward(
            d_model=self.d_model, d_ff=self.d_ff, p_dropout=self.p_dropout
        )
        self.norm1 = hk.LayerNorm(
            axis=-1, param_axis=-1, create_scale=True, create_offset=True
        )
        self.norm2 = hk.LayerNorm(
            axis=-1, param_axis=-1, create_scale=True, create_offset=True
        )
        self.norm3 = hk.LayerNorm(
            axis=-1, param_axis=-1, create_scale=True, create_offset=True
        )

    def __call__(self, x, memory, src_mask, tgt_mask, is_train):
        """
        The forward pass of the decoder block.
        
        :param x: the input sequence for the decoder block.
        :param memory: the memory from the encoder.
        :param src_mask: the mask for the src sequence.
        :param tgt_mask: the mask for the tgt sequence.
        :param is_train: boolean flag to indicate training mode.
        :return: the output of the decoder block.
        """
        # self-attention sub-layer
        sub_x, _ = self.self_attn(x, x, x, tgt_mask)
        if is_train:
            sub_x = hk.dropout(hk.next_rng_key(), self.p_dropout, sub_x)
        x = self.norm1(x + sub_x)  # residual conn
        # cross-attention sub-layer
        sub_x, _ = self.cross_attn(x, memory, memory, src_mask)
        if is_train:
            sub_x = hk.dropout(hk.next_rng_key(), self.p_dropout, sub_x)
        x = self.norm2(x + sub_x)
        # feedforward sub-layer
        sub_x = self.ff(x, is_train=is_train)
        if is_train:
            sub_x = hk.dropout(hk.next_rng_key(), self.p_dropout, sub_x)
        x = self.norm3(x + sub_x)

        return x
```

```python
class TransformerDecoder(hk.Module):
    """
    The Transformer decoder model.
    
    :param num_layers: number of decoder layers.
    :param num_heads: number of attention heads.
    :param d_model: dimension of the model.
    :param d_ff: dimension of the feedforward network model.
    :param p_dropout: dropout rate.
    """
    
    def __init__(self, num_layers, num_heads, d_model, d_ff, p_dropout, name=None):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.p_dropout = p_dropout

        self.layers = [
            DecoderBlock(self.d_model, self.num_heads, self.d_ff, self.p_dropout)
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, memory, src_mask, tgt_mask, is_train):
        """
        The forward pass of the decoder.
        
        :param x: the input sequence for the decoder.
        :param memory: the memory from the encoder.
        :param src_mask: the mask for the src sequence.
        :param tgt_mask: the mask for the tgt sequence.
        :param is_train: boolean flag to indicate training mode.
        :return: the output of the transformer decoder.
        """
        for l in self.layers:
            x = l(x, memory, src_mask, tgt_mask, is_train)
        return x
```

```python
class Transformer(hk.Module):
    """
    Complete Transformer model including encoder and decoder.
    
    :param d_model: dimension of the model.
    :param d_ff: dimension of the feedforward network model.
    :param src_vocab_size: size of the source vocabulary.
    :param tgt_vocab_size: size of the target vocabulary.
    :param num_layers: number of encoder and decoder layers.
    :param num_heads: number of attention heads.
    :param p_dropout: dropout rate.
    :param max_seq_len: maximum sequence length.
    """
    def __init__(
        self,
        d_model,
        d_ff,
        src_vocab_size,
        tgt_vocab_size,
        num_layers,
        num_heads,
        p_dropout,
        max_seq_len,
        name=None,
        tie_embeddings=False,
    ):
        
        super().__init__(name)

        self.d_model = d_model
        self.d_ff = d_ff
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.max_seq_len = max_seq_len

        self.src_emb = Embeddings(d_model, src_vocab_size)
        if tie_embeddings:
            self.tgt_emb = self.src_emb
        else:
            self.tgt_emb = Embeddings(d_model, tgt_vocab_size)
        self.encoder = TransformerEncoder(
            num_layers, num_heads, d_model, d_ff, p_dropout
        )
        self.decoder = TransformerDecoder(
            num_layers, num_heads, d_model, d_ff, p_dropout
        )

    def encode(self, src, src_mask, is_train):
        """
        The forward pass for the encoder.
        
        :param src: the source sequence.
        :param src_mask: the mask for the src sequence.
        :param is_train: boolean flag to indicate training mode.
        :return: the encoded sequence.
        """
        pe = PositionalEncoding(self.d_model, self.max_seq_len, self.p_dropout)
        src = self.src_emb(src)
        src = src[None, :, :] if len(src.shape) == 2 else src
        src = pe(src, is_train=is_train)
        return self.encoder(src, src_mask, is_train)

    def decode(self, memory, src_mask, tgt, tgt_mask, is_train):
        """
        The forward pass for the decoder.
        
        :param memory: the memory from the encoder.
        :param src_mask: the mask for the src sequence.
        :param tgt: the target sequence.
        :param tgt_mask: the mask for the tgt sequence.
        :param is_train: boolean flag to indicate training mode.
        :return: the output of the decoder.
        """
        pe = PositionalEncoding(self.d_model, self.max_seq_len, self.p_dropout)
        tgt = self.tgt_emb(tgt)
        tgt = tgt[None, :, :] if len(tgt.shape) == 2 else tgt
        tgt = pe(tgt, is_train=is_train)
        return self.decoder(tgt, memory, src_mask, tgt_mask, is_train)

    def __call__(self, src, src_mask, tgt, tgt_mask, is_train):
        """
        The forward pass of the whole transformer model.
        
        :param src: the source sequence.
        :param src_mask: the mask for the src sequence.
        :param tgt: the target sequence.
        :param tgt_mask: the mask for the tgt sequence.
        :param is_train: boolean flag to indicate training mode.
        :return: the output of the transformer model (encoder + decoder).
        """
        memory = self.encode(src, src_mask, is_train)
        return self.decode(memory, src_mask, tgt, tgt_mask, is_train)
```