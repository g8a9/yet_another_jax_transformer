# Combining all together: the Transformer Encoder

The Transformer encoder is composed of multiple *encoder blocks*. Each of these blocks comprises two sub-layers: a *multi-head self-attention layer*, and a *feed-forward network*. There is also a residual connection around each sub-layer, followed by *layer normalization*. See the Figure above for a detailed diagram of a single encoder block.

## Feed Forward Sublayer

This sublayer is composed of a fully-connected feed-forward network. The main idea is to learn a linear transformation of the hidden representation of the previous layer. This layer has an inner hidden layer of size `d_ff`, and an inner activation function (e.g., ReLU). The `PositionwiseFeedForward` class below implements this sub-layer. It is initialized using the parameters:

- `d_model`: size of the hidden representation of the input.
- `d_ff`: inner size of the hidden layer.
- `p_dropout`: dropout probability (dropout will be applied during training).

The `PositionwiseFeedForward` class implements the `__call__` method. It takes as input the previous layer's hidden representation and returns the current layer's hidden representation by applying the fully-connected network.

```python
class PositionwiseFeedForward(hk.Module):
    """
    This class is used to create a position-wise feed-forward network.
    :param d_model: The size of the embedding vector.
    :param d_ff: The size of the hidden layer.
    :param p_dropout: The dropout probability.
    """
    def __init__(self, d_model: int, d_ff: int, p_dropout: float = 0.1, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.d_ff = d_ff
        self.p_dropout = p_dropout

        self.w_1 = hk.Linear(self.d_ff)
        self.w_2 = hk.Linear(self.d_model)

    def __call__(self, x, is_train=True):
        """
        :param x: The input sequence.
        :param is_train: Whether the model is in training mode.
        :return: The output of the position-wise feed-forward network.
        """
        x = jax.nn.relu(self.w_1(x))
        if is_train:
            x = hk.dropout(hk.next_rng_key(), self.p_dropout, x)

        x = self.w_2(x)
        return x
```

In the last cell, we used `hk.next_rng_key()`. You can call this haiku utility function 
**only from within a haiku.Module** to get a new PNRGenerator key.

## Encoder Block

The `EncoderBlock` contains all the components of a single encoder block. It is initialized using the parameters:

- `d_model`: the size of the hidden representation of the input.
- `num_heads`: number of heads in the multi-headed attention layer.
- `d_ff`: the inner size of the hidden layer of the position-wise feed-forward sub-layer.
- `p_dropout`: dropout probability (dropout will be applied during training).

It applies the two sub-layers: the multi-head self-attention layer and the position-wise feed-forward sub-layer.
The `__init__` method is used to initialize the parameters of the encoder block, while the `__call__` method applies the encoder block to an input.

```python
class EncoderBlock(hk.Module):
    """
    This class is used to create an encoder block.

    :param d_model: The size of the embedding vector.
    :param num_heads: The number of attention heads.
    :param d_ff: The size of the hidden layer.
    :param p_dropout: The dropout probability.
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

    def __call__(self, x, mask=None, is_train=True):
        """
        It applies the encoder block to the input sequence.

        :param x: The input sequence.
        :param mask: The mask to be applied to the self-attention layer.
        :param is_train: Whether the model is in training mode.
        :return: The output of the encoder block, which is the updated input sequence.
        """
        d_rate = self.p_dropout if is_train else 0.0

        # attention sub-layer
        sub_x, _ = self.self_attn(x, x, x, mask=mask)
        if is_train:
            sub_x = hk.dropout(hk.next_rng_key(), self.p_dropout, sub_x)
        x = self.norm1(x + sub_x)  # residual conn

        # feedforward sub-layer
        sub_x = self.ff(x, is_train=is_train)
        if is_train:
            sub_x = hk.dropout(hk.next_rng_key(), self.p_dropout, sub_x)
        x = self.norm2(x + sub_x)  # sub_x

        return x
```

Let's do our usual test.

```python
"""Testing the Encoder block"""

bs = 2
seq_len = 12
d_model = 64
num_heads = 8
d_ff = 128


@hk.transform
def enc_blk(x, mask, is_train):
    bl = EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, p_dropout=0.1)
    return bl(x, mask, is_train)


## Test EncoderBlock implementation
# Example features as input
rng_key = next(rng_iter)
x = jax.random.normal(rng_key, (bs, seq_len, d_model))
mask = jax.random.randint(rng, (bs, 1, seq_len), minval=0, maxval=2)

# Initialize parameters of encoder block with random key and inputs
params = enc_blk.init(rng=rng_key, x=x, mask=mask, is_train=True)

# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
out = enc_blk.apply(rng=rng_key, params=params, x=x, mask=mask, is_train=True)
print("Out", out.shape)

del enc_blk, params
```

## Transformer Encoder

As introduced in the previous sections, the Transformer encoder is composed of multiple *encoder blocks*. The `TransformerEncoder` class below implements it by stacking $N$ `EncoderBlock`s, where $N$ is the number of stacked encoder blocks.

This class inputs the same set of parameters as the `EncoderBlock` class and adds the parameter `num_layers` to specify the number of stacked encoder blocks.

```python
class TransformerEncoder(hk.Module):
    """
    This class is used to create a transformer encoder.
    :param num_layers: The number of encoder blocks.
    :param num_heads: The number of attention heads.
    :param d_model: The size of the embedding vector.
    :param d_ff: The size of the hidden layer.
    :param p_dropout: The dropout probability.
    """

    def __init__(self, num_layers, num_heads, d_model, d_ff, p_dropout, name=None):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.p_dropout = p_dropout

        self.layers = [
            EncoderBlock(self.d_model, self.num_heads, self.d_ff, self.p_dropout)
            for _ in range(self.num_layers)
        ]

    def __call__(self, x: List[int], mask=None, is_train=True):
        """
        It applies the transformer encoder to the input sequence.
        :param x: The input sequence.
        :param mask: The mask to be applied to the self-attention layer.
        :param is_train: Whether the model is in training mode.
        :return: The final output of the encoder that contains the last encoder block output.
        """
        for l in self.layers:
            x = l(x, mask=mask, is_train=is_train)
        return x
```

Let's run our encoder block.

```python
"""Testing the Transformer Encoder"""
bs = 2
seq_len = 12
d_model = 64
num_heads = 8
d_ff = 128
num_layers = 6
p_dropout = 0.1


@hk.transform
def transformer_encoder(x, mask, is_train):
    enc = TransformerEncoder(num_layers, num_heads, d_model, d_ff, p_dropout, "t_enc")
    return enc(x, mask, is_train)

## Test TransformerEncoder implementation
# Example features as input
rng_key = next(rng_iter)
x = jax.random.normal(rng_key, (bs, seq_len, d_model))
mask = jax.random.randint(rng, (bs, 1, seq_len), minval=0, maxval=2)

# Initialize parameters of transformer with random key and inputs
params = transformer_encoder.init(rng=rng_key, x=x, mask=mask, is_train=True)

# Apply transformer with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
out = transformer_encoder.apply(
    rng=rng_key, params=params, x=x, mask=mask, is_train=True
)
print(out.shape)

del params, transformer_encoder
```