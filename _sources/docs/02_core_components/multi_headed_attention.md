# The Multi-Headed Attention

The scaled dot product attention allows an element of the sequence to attend to any other element. However, the scaled dot product attention does not allow the element to focus on multiple aspects of the sequence simultaneously.
A solution for this is to use multiple attention heads.

Indeed, the first unit of the encoder applies a *multi-headed self-attention*, meaning that i) words *mix and align among themselves* (self-attention) and ii) multiple, different alignments are learned at once (multi-headed) -- each alignment is imputed to one *attention head*.

This simple learning paradigm -- based on mixing and aligning words in sentences -- paired with a linguistically founded training objective enables the best performing language models.

With the multi-headed attention, we have $h$ attention heads, where each attention head is a linear projection of the sequence $Q$, $K$, and $V$:

$$
attention(Q, K, V) = \text{concat}(head_1,...,head_h)W^O
$$

$$
head_i = attention(QW^Q_i, KW^K_i, VW^V_i)
$$

Where $W^Q_i \in \mathbb{R}^{d_q \times d_k/h}$, $W^K_i \in \mathbb{R}^{d_k \times d_k/h}$, $W^V_i \in \mathbb{R}^{d_v \times d_v/h}$, and $W^O \in \mathbb{R}^{hd_v \times d_v}$. Note that the $d_k$ and $d_v$ have the same dimension, so the $d_v/h$ is the same as the $d_k/h$.

While implementing multi-headed attention, we implement the linear projection $Q$, $K$, and $V$ with matrix multiplication and then split the result into $h$ heads. We then apply the scaled dot product for each attention head independently and concatenate the results.

We provide you with Multi-Headed Attention in the form of a Haiku Module. Take your time to go through the class and understand the main components.

```python
class MultiheadAttention(hk.Module):
    """
    Multihead Attention module.
    :param d_model: dimension of the model
    :param num_heads: number of heads
    """
    
    def __init__(self, d_model: int, num_heads: int, name=None):
        super().__init__(name=name)

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.lin_projs = [hk.Linear(self.d_model) for _ in range(4)]

    def __call__(self, q, k, v, mask=None):
        """
        Perform Multi-Headed Attention.

        :param q: queries tensor (B,...,S,d_model)
        :param k: keys tensor (same)
        :param v: values tensor (same)
        :param mask: mask tensor (broadcastable to: B,...,S,S)
        """
        batch_size, seq_length, d_model = q.shape # (B,S,d_k)

        q, k, v = [
            lin_p(t).reshape(batch_size, -1, self.num_heads, self.d_k).swapaxes(1, 2)
            for lin_p, t in zip(self.lin_projs, (q, k, v))
        ]  # (B,h,S,d_k)

        if mask is not None:
            mask = jnp.expand_dims(mask, 1)  # expand to (B,h,...)

        values, attention = scaled_dot_product(q, k, v, mask=mask)  # (B,h,S,d_k)
        values = values.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, d_model)  # concat heads
        return self.lin_projs[-1](values), attention
```

What you should have noticed (and we are sure you did):

- following the original paper, we set a $d_{model}$ and a $d_k = d_{model} / h$;
- queries, keys, and values have three different projections;
- we do not implement linear projection by ourselves but use [hk.Linear](https://dm-haiku.readthedocs.io/en/latest/api.html#linear) that needs the desired output dimension in the constructor;
- each query/key/value vector is 1) linearly projected 2) reshaped to h heads and a size of $d_k$ 3) swapped axes such that the head axis is at position 1;
- we add a dimension to the mask corresponding to the attention heads, such that the model will mask every attention head equally (equal masks are expected in Encoders, while we will see that there will be different masks in Decoders).  

Again, let's test our implementation.

```python
""" Test MultiheadAttention implementation """
bs = 2
seq_len = 12
d_model = 64
num_heads = 8


def test_mha(q, k, v, mask=None):
    mha = MultiheadAttention(d_model, num_heads, name="mha")
    return mha(q, k, v, mask)


mha = hk.without_apply_rng(hk.transform(test_mha))

# Example features as input
q, k, v = jax.random.normal(next(rng_iter), (3, bs, seq_len, d_model))
mask = jax.random.randint(rng, (bs, 1, seq_len), minval=0, maxval=2)

# Initialize parameters of attention with random key and inputs
params = mha.init(next(rng_iter), q, k, v, mask)

# Apply attention with parameters on the inputs
out, attn = mha.apply(params=params, q=q, k=k, v=v, mask=mask)
print("Out", out.shape, "Attention", attn.shape)
del mha, params
```

In the last cell, we used:
- `hk.without_apply_rng`: it is a wrapper that let us apply a function without passing `rng` as an argument. As long as the function is not actually using random numbers during computation, we can use `without_apply_rng`.
- `hk.transform`: it is a very handy module in Haiku (also used as a decorator: `@hk.transform`) that allows the definition of a pure function. From the original Haiku documentation:

```{note}
The transform function allows you to write neural network functions that rely on parameters (...) without requiring you to explicitly write the boilerplate for initialising those parameters. `transform` does this by transforming the function into a pair of functions that are pure (as required by JAX) init and apply.
```