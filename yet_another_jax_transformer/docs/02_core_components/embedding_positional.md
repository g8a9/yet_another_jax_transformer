# Turning Tokens into Vectors: Embeddings and Positional Encoding 

The Transformer takes a sequence of words (or tokens) represented by dense vectors as input. These vectors are called *embeddings*, and their role is to map words (tokens) into a continuous vector space. The model's input is thus a sequence of vectors obtained by *looking up* the embedding of the corresponding words (tokens) in the vocabulary.

## Embedding Layer

As an exercise, we ask you to write an `Embeddings` class that takes as input the dimension of the embeddings' vectors for the model (`d_model`) and the size of the vocabulary (`vocab_size`). Your class should implement the `__call__` method that takes as input a sequence of integers, each integer corresponding to a word (token) in the vocabulary, and outputs a sequence of vectors, each vector corresponding to the embedding of the corresponding word (token).

```python
class Embeddings(hk.Module):
    """
    This class is used to create an embedding matrix for a given vocabulary size.
    :param d_model: The size of the embedding vector.
    :param vocab_size: The size of the vocabulary.
    """
    def __init__(self, d_model, vocab_size, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = hk.Embed(self.vocab_size, self.d_model)

    def __call__(self, x):
        """
        :param x: The input sequence.
        :return: The embedding matrix.
        """
        return self.embeddings(x) * jnp.sqrt(self.d_model)
```

We can test it right away.

```python
""" Test Embeddings implementation """

bs = 2
seq_len = 12
d_model = 64
num_heads = 8
vocab_size = 100

test_emb = lambda inputs: Embeddings(d_model, vocab_size)(inputs)
emb = hk.without_apply_rng(hk.transform(test_emb))

# example features as input
inputs = jax.random.randint(next(rng_iter), (4, 3), 0, 5)
params = emb.init(next(rng_iter), inputs)
out = emb.apply(params=params, inputs=inputs)
print("Out", out.shape)
del emb, params
```

## Positional Encoding

The Transformer model does not use recurrent or convolutional layers in the encoder/decoder of the model (only attention mechanisms). However, this also has a drawback: since the model has no memory (no recurrent/convolutional layers), it can not take into account the *order* of the sequence elements. The position of words in the sequence is thus not encoded explicitly by the model.

As a solution to this issue, the original Transformer model uses a *positional encoding* scheme to represent the position of each element in the sequence. The positional encoding is added to the token embeddings of each element. Following the original paper, positional encodings are generated with multiple sinusoidal functions with varying frequencies.

Positional encoding is defined as:

$$\text{PE}(pos, 2i) = \sin \left( \frac{pos}{1000^{2i/d_{\text{model}}}} \right)$$
$$\text{PE}(pos, 2i+1) = \cos \left( \frac{pos}{1000^{2i/d_{\text{model}}}} \right)$$

where $pos$ is the position of the element in the sequence, $d_{\text{model}}$ is the model's embedding dimension, and $i$ is the index of the position vector. Note that this is not a learned parameter; the values are pre-computed and added to the token embeddings at the beginning of the forward pass.

Note that, we can optionally apply dropout to the positional encodings during training, thus providing additional regularization for the model.

📚 **Resources**

- Detailed explanation with visual aids: [Understanding Positional Encoding in Transformers](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers)

```python
class PositionalEncoding(hk.Module):
    """
    This class is used to add positional encoding to the input sequence.
    :param d_model: The size of the embedding vector.
    :param max_len: The maximum length of the input sequence.
    :param p_dropout: The dropout probability.
    """
    def __init__(self, d_model: int, max_len: int, p_dropout: float = 0.1, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_len = max_len
        self.p_dropout = p_dropout

        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
        )
        pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x, is_train=True):
        """
        :param x: The input sequence.
        :param is_train: Whether the model is in training mode.
        :return: The input sequence with positional encoding.
        """
        x = x + self.pe[:, : x.shape[1]]
        if is_train:
            return hk.dropout(hk.next_rng_key(), self.p_dropout, x)
        else:
            return x
```