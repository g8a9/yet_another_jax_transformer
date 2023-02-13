# Attention Mechanism in the Transformer

It is safe to say that the attention mechanism lies at the core of *all* best-performing language models. This simple alignment algorithm is the foundation of how we model natural language today.

Before reviewing Attention in Transformer, we provide the intuition using influencers and dress styles.

```{note}
Fashion trends change rapidly. Harry knows that and tries to keep his wardrobe ready. Every season he goes over the social profiles of his favorite fashion influencers to look for ideas. Harry finds nice shirts in profile 1, suitable shoes in profile 2, nothing exciting in profile 3, and so on. From each influencer, he chooses part of the outfit for the upcoming season. In a sense, Harry **aligns** his preferences with social profiles and **mixes** different styles, following his intuition on what is best for his final goal -- we do not know Harry. Maybe he is trying to be a famous influencer himself.
```

Transformers learn word representations similarly. Each word is a **query** (Harry's outfit) whose representation is updated in alignment with a set of other words (the influencers' profiles), the **keys**, mixing some of their **values** (the influencers' products). Also, some training objective (Harry's dream of becoming an influencer) drives the process.

Let's define our queries, keys, and values.

## Scaled Dot Product Attention

Scaled Dot Product attention is the attention mechanism used in Transformer. A query and a key-value pair are used to compute the attention. First, queries ($Q$) and keys ($K$) are multiplied; then, softmax is applied to the result to obtain the attention scores. Finally, values ($V$) are multiplied by the attention scores to get the final representation of the sequence.

The scaled dot product attention is defined as:

$$
attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

Where $d_k$ is a constant scalar. In the original paper, $d_k$ corresponds to the dimension of the query/key/value (they all share the same dimension).

📣 📣 📣

It is a good place to pause and talk about vectors and dimensions. Along the notebook, you will work with JAX arrays (vectors) with one or more dimensions and operators (either mathematical operations or explicit function calls) that reshape them. Also, several operators use [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). You are very well encouraged to refreshen that logic before the start.

Since it is easy to lose track of the dimensions involved, we will sometime specify the expected shape as a code comment.
Unless specified, we will use `B` for the batch size, `S` for the sequence length, and `h` for the number of attention heads. When we use an inline comment, the shape refers to the resulting value of the statement, e.g.:

`scores = ...  # (B,...,S,S)`

means that, after the execution, `scores` will have a shape of B, any random number of dimensions, and will end with the two final dimensions with size S. 

📣 📣 📣

Let's now see the JAX implementation of the Scaled Dot Product Attention (Equation above). You will notice that the function accepts a `mask` parameter. The mask allows us to *ignore* some portion of the sequence (typically, padding tokens if present).

```python
def scaled_dot_product(q, k, v, mask=None):
    """
    Perform Scaled Dot Product Attention.

    :param q: queries tensor (shape: B,...,S,d_k)
    :param k: keys tensor (shape: B,...,S,d_k)
    :param v: values tensor (shape: B,...,S,d_k)
    :param mask: mask tensor (shape broadcastable to: B,...,S,S)
    :return: attention output (shape: B,...,S,d_k), attention_weights (B,...,S,S)
    """
    d_k = q.shape[-1]
    scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(d_k)  # (B,...,S,S)

    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)

    attention_weights = jax.nn.softmax(scores, axis=-1)
    values = jnp.matmul(attention_weights, v)
    return values, attention_weights
```

We can now test our code.

```python
# Testing Scaled Dot Product
bs = 2
seq_len, d_k = 3, 4
rng = next(rng_iter)

q, k, v = jax.random.normal(rng, (3, bs, seq_len, d_k))
mask = jax.random.randint(rng, (bs, 1, seq_len), minval=0, maxval=2)
values, attention = scaled_dot_product(q, k, v, mask)

print("Values\n", values, values.shape)  # result should be (B,S,d_k)
print("Attention\n", attention, attention.shape)  # result should be (B,S,S)
```