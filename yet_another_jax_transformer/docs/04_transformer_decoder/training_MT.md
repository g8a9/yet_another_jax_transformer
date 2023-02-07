# Training a Neural Machine Translation Model 🇬🇧 -> 🇮🇹

## Defining the model transformation \[EXERCISE 📝\]

We can now define the Transformer model that will be used to translate from English to Italian. We also define the criterion (loss function) we can use to train the MT model. Similarly to the Encoder model, we will use the Cross-Entropy loss, but we need to compute it across all target words.

```python
@hk.transform
def mt_model(src, src_mask, tgt, tgt_mask, is_train=True):
    """
    The machine translation model that relies on the encoder and decoder defined above.

    :param src: source sequences
    :param src_mask: source mask
    :param tgt: target sequences
    :param tgt_mask: target mask
    :param is_train: whether the model is in training mode or not
    :return: logits
    """
    model = Transformer(
        d_model=D_MODEL,
        d_ff=D_FF,
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        p_dropout=P_DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        tie_embeddings=True,
    )

    output_embs = model(src, src_mask, tgt, tgt_mask, is_train=is_train)

    # final decoder
    out = hk.Linear(VOCAB_SIZE, name="decoder_final_linear")(output_embs)  # logits
    out = jax.nn.log_softmax(out)
    return out

def prepare_sample(tokenizer, src_text, tgt_text=None, max_seq_len=MAX_SEQ_LEN):
    """
    Prepare a sample for the model. 
    This function encode the source and target sequences to generate the sentence pair.
    It also generates the attention masks for the source and target sequences.
    
    :param tokenizer: the tokenizer to use
    :param src_text: the source text
    :param tgt_text: the target text
    :param max_seq_len: the maximum sequence length
    :return: a tuple of (src_enc, src_mask, tgt_enc, tgt_mask) if tgt_text is not None, otherwise (src_enc, src_mask)
    """

    src_enc = tokenizer.encode(src_text, add_special_tokens=False)
    src = jnp.array([src_enc.ids])
    src_mask = jnp.expand_dims(jnp.array([src_enc.attention_mask]), 1)

    item = (src, src_mask)

    if tgt_text is not None:
        tgt_enc = tokenizer.encode(tgt_text, add_special_tokens=True)
        tgt = jnp.array([tgt_enc.ids])
        tgt_mask = subsequent_mask(max_seq_len)
        item += (tgt, tgt_mask)

    return item
```

Let's test the model and the loss function.

```python
# testing the MT model
src, src_mask, tgt, tgt_mask = prepare_sample(
    mt_tokenizer, "Hello my friend", "Ciao amico mio"
)

rng_key = next(rng_iter)
params = mt_model.init(rng_key, src, src_mask, tgt, tgt_mask, True)
logits = mt_model.apply(
    params,
    rng=rng_key,
    src=src,
    src_mask=src_mask,
    tgt=tgt,
    tgt_mask=tgt_mask,
    is_train=False,
)
print("Logits shape", logits.shape)
```

```python
# testing loss functions
counter = 0
for batch in train_loader_mt:
    out = mt_model.apply(
        params=params,
        rng=rng_key,
        src=batch["src"],
        src_mask=batch["src_mask"],
        tgt=batch["tgt"],
        tgt_mask=batch["tgt_mask"],
        is_train=True,
    )

    labels = batch["labels"]
    loss = optax.softmax_cross_entropy_with_integer_labels(out, labels)
    loss = jnp.where(labels != PAD_ID, loss, 0.0)
    not_pad_count = (labels != PAD_ID).sum()

    print(loss.sum() / not_pad_count)

    if counter >= 1:
        break
    else:
        counter += 1
```

## Setup the training loop

```python
EPOCHS = 5  # @param {type:"number"}
EVAL_STEPS = 500  # @param {type:"number"}
LOG_STEPS = 200

# Initialise network and optimiser; note we draw an input to get shapes.
sample = proc_datasets["train"][0]
src, src_mask, tgt = map(
    jnp.array,
    (
        sample["src_ids"],
        sample["src_mask"],
        sample["tgt_ids"],
    ),
)
tgt_mask = subsequent_mask(MAX_SEQ_LEN)

rng_key = next(rng_iter)
init_params = mt_model.init(rng_key, src, src_mask, tgt, tgt_mask, True)

# We use learning rate scheduling / annealing
total_steps = EPOCHS * len(train_loader_mt)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6,
    peak_value=LEARNING_RATE,
    warmup_steps=int(0.1 * total_steps),
    decay_steps=total_steps,
    end_value=1e-6,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP_VALUE),
    optax.adam(learning_rate=LEARNING_RATE),
)
init_opt_state = optimizer.init(init_params)

# initialize the training state class
state = TrainingState(init_params, init_opt_state)
```

```python
def loss_fn_mt(params: hk.Params, batch, rng) -> jnp.ndarray:
    """
    The loss function for the machine translation model.
    The loss is computed as the sum of the cross entropy loss for each token in the target sequence.
    
    :param params: the model parameters
    :param batch: the batch of data
    :param rng: the random number generator
    :return: the loss value
    """

    """EXERCISE"""
    logits = mt_model.apply(
        params=params,
        rng=rng,
        src=batch["src"],
        src_mask=batch["src_mask"],
        tgt=batch["tgt"],
        tgt_mask=batch["tgt_mask"],
        is_train=True,
    )

    labels = batch["labels"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = jnp.where(labels != PAD_ID, loss, 0.0)
    not_pad_count = (labels != PAD_ID).sum()
    return loss.sum() / not_pad_count


@jax.jit
def train_step_mt(state, batch, rng_key) -> TrainingState:
    """
    The training step for the machine translation model.

    :param state: the state of the training
    :param batch: the batch of data
    :param rng_key: the random number generator
    :return: the new training state, the metrics (training loss) and the random number generator
    """
    rng_key, rng = jax.random.split(rng_key)

    loss_and_grad_fn = jax.value_and_grad(loss_fn_mt)
    loss, grads = loss_and_grad_fn(state.params, batch, rng_key)

    updates, opt_state = optimizer.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(params, opt_state)
    metrics = {"train_loss": loss}

    return new_state, metrics, rng_key


@jax.jit
def deterministic_forward(
    params: hk.Params, src, src_mask, tgt, tgt_mask
) -> jnp.ndarray:
    """
    The deterministic forward pass for the machine translation model.
    It leverages without_apply_rng to avoid the need for a random number generator.
    
    :param params: the model parameters
    :param src: the source sequences
    :param src_mask: the source mask
    :param tgt: the target sequences
    :param tgt_mask: the target mask
    :return: the logits
    """
    return hk.without_apply_rng(mt_model).apply(
        params=params,
        is_train=False,
        src=src,
        src_mask=src_mask,
        tgt=tgt,
        tgt_mask=tgt_mask,
    )


@jax.jit
def eval_step_mt(params: hk.Params, batch) -> jnp.ndarray:
    """
    The evaluation step for the machine translation model.
    :param params: the model parameters
    :param batch: the batch of data
    :return: the evaluation loss
    """
    logits = deterministic_forward(
        params, batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"]
    )
    labels = batch["labels"]

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = jnp.where(labels != PAD_ID, loss, 0.0)
    not_pad_count = (labels != PAD_ID).sum()
    return loss.sum() / not_pad_count
```

## Training and evaluation loop

```python
writer = SummaryWriter()
pbar = tqdm(desc="Train step", total=EPOCHS * len(train_loader_mt))
step = 0
loop_metrics = {"train_loss": None, "eval_loss": None}
best_eval_loss = float("inf")

for epoch in range(EPOCHS):

    for batch in train_loader_mt:

        state, metrics, rng_key = train_step_mt(state, batch, rng_key)
        loop_metrics.update(metrics)
        pbar.update(1)
        step += 1

        if step % EVAL_STEPS == 0:
            ebar = tqdm(desc="Eval step", total=len(valid_loader_mt), leave=False)

            losses = list()
            for batch in valid_loader_mt:
                loss = eval_step_mt(state.params, batch)
                losses.append(loss)
                ebar.update(1)
            ebar.close()

            eval_loss = jnp.array(losses).mean()
            loop_metrics["eval_loss"] = eval_loss

            writer.add_scalar("Loss/valid", loop_metrics["eval_loss"].item(), step)

            if eval_loss.item() < best_eval_loss:
                best_eval_loss = eval_loss.item()
                # Save the params training state (and params) to disk
                with open(f"mt_train_state_{step}.pkl", "wb") as fp:
                    pickle.dump(state, fp)

        if step % LOG_STEPS == 0:
            writer.add_scalar("Loss/train", loop_metrics["train_loss"].item(), step)
            writer.add_scalar("lr/train", schedule(step).item(), step)
            writer.add_scalar("epoch/train", epoch, step)

        pbar.set_postfix(loop_metrics)

pbar.close()
```

## Implement Greedy Decoding

We iteratively process the sequence through the encoder to generate the output sequence. Specifically, we will use **greedy decoding**: we take the token with the highest log-likelihood (logit) at each step to generate the complete output sequence.

Decoding strategies are a broad research topic that we are touching only on the most naive approach. For a basic introduction to other generation strategies, please refer to [this blog post](https://huggingface.co/blog/how-to-generate).

**🤔 Switching to Europarl?**

By now, you should have an MT model trained on TatoEBA. We trained for you a similar model on Europarl.
You can now decide to continue with your model or load our pretrained.

If you want to load the Europarl model, run the cell below to load the checkpoint and tokenizer, and set the hyperparameters accordingly (we will download the checkpoint saved after 596000 steps. Feel free to choose any other checkpoint in the folder).

```bash
mkdir europarl_pretrained
curl -L https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/models/europarl/train_state_596000.pkl -o europarl_pretrained/state.pkl
curl -L https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/models/europarl/config.json -o europarl_pretrained/config.json
curl -L https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/models/europarl/tokenizer.json -o europarl_pretrained/tokenizer.json
```

```python
checkpoint_file = "./europarl_pretrained/state.pkl"
tokenizer_file = "./europarl_pretrained/tokenizer.json"

NUM_LAYERS = 6
NUM_HEADS = 8
D_MODEL = 512
D_FF = 1024
MAX_SEQ_LEN = 256
VOCAB_SIZE = 20_000

with open(checkpoint_file, "rb") as fp:
    state = pickle.load(fp)

tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
tokenizer.enable_truncation(MAX_SEQ_LEN)
```

Let's now implement the actual greedy deconding function.

```python
def translate(params, query, tokenizer, show_progress=False):
    """
    Translate a query using the machine translation model.
    This function uses the greedy decoding strategy.

    :param params: the model parameters
    :param query: the query to translate
    :param tokenizer: the tokenizer
    :param show_progress: whether to show the progress of the translation
    :return: the translated query
    """
    model = hk.without_apply_rng(mt_model)
    src, src_mask = prepare_sample(src_text=query, tokenizer=tokenizer)

    tgt = jnp.full((1, 1), tokenizer.token_to_id("[BOS]"), dtype=src.dtype)

    for i in tqdm(range(MAX_SEQ_LEN - 1), desc="Decoding", disable=not show_progress):
        logits = deterministic_forward(
            params, src, src_mask, tgt, subsequent_mask(tgt.shape[1])
        )
        next_word = logits[0, i, :].argmax()
        tgt = jnp.concatenate(
            [tgt, jnp.full((1, 1), next_word, dtype=src.dtype)], axis=-1
        )
        if next_word == tokenizer.token_to_id("[EOS]"):
            break

    return tokenizer.decode(tgt[0]).replace(" ##", "")
```

Here is a simple translation.

```python
query = "The doctor is ready for the operation."
tgt = translate(state.params, query, tokenizer, show_progress=True)
print(tgt)
```