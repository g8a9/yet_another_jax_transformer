# 🚀 Training your First Language Model

Before starting, let us recap the object required to pre-train the Transformer encoder:

- ✅ **Model**: Transformer Encoder (which we already implemented)
- 📝 **Dataset**: As the training objective is token-level MLM, we can use any text corpus. In our case, we will use a toy dataset derived from Tatoeba.
- 📝 **Tokenizer**: We need a tokenizer that takes a string and returns a list of tokens. It is in charge of splitting the input text into tokens and mapping each token to a unique integer index. We are going to use the `BPE` [(byte pair encoding)](https://huggingface.co/course/chapter6/5) tokenizer provided by the [tokenizers](https://huggingface.co/docs/tokenizers/index) library.
- 📝 **Training loop**: We need a training loop that iterates over the dataset, computes the loss, back-propagates the gradients, and updates the parameters.

```python
# some global variables
BATCH_SIZE = 64
MASK_PROBABILITY = 0.15
NUM_LAYERS = 6
NUM_HEADS = 8
D_MODEL = 128
D_FF = 256
P_DROPOUT = 0.1
MAX_SEQ_LEN = 128
VOCAB_SIZE = 25000
LEARNING_RATE = 3e-4
GRAD_CLIP_VALUE = 1
```

## Tatoeba dataset

[Tatoeba](https://tatoeba.org/) is an open and collaborative platform for collecting translations in different languages. It is an excellent resource for machine translation tasks. 


![image](https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/tatoeba_example.png)

For our toy example, we will use a small subset of the Tatoeba dataset consisting of aligned sentence pairs in Italian and English.

We only need the English sentences from the dataset to train our Transformer encoder. The English-Italian sentence pairs will be used in the next section when we train a Transformer encoder-decoder.

You can download the dataset by running the following cell.

```bash
curl -LO https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/it-en.tsv
```

It will download a `tsv` file named `it-en.tsv`. We can load it using `pandas` and collect only the English sentences we will use for our MLM pre-training.

```python
df = pd.read_csv(
    "it-en.tsv", sep="\t", header=0, names=["id_it", "sent_it", "id_en", "sent_en"]
)
df = df.dropna()

# We will use english sentences to train our encoder with MLM
en_sentences = df["sent_en"].drop_duplicates()
print(f"Unique English sentences: {len(en_sentences)}")
print("Samples:\n", en_sentences[:5])
```

## Training a BPE Tokenizer

Before starting training our Transformer model, we need to train a tokenizer that we will use to split the input text into tokens. The *tokenizers* library provides many tokenizers, including the `BPE` tokenizer we will use.

BPE tokenization involves the following steps:

1. The corpus is split to obtain a set of characters.
2. Pairs of characters are combined to form sub-words according to some frequency metric.
3. Process at 2. is repeated until the condition on the maximum number of sub-words in the vocabulary is met.
4. The vocabulary is generated by taking the final set of sub-words.

We need a `VOCAB_SIZE` parameter that defines our vocabulary's maximum capacity (number of tokens). We will also leverage another global variable, `MAX_SEQ_LENGTH`, that sets the maximum sentence length to a fixed number of tokens.

🚨🚨🚨 

We usually refer to **tokens** instead of words when training NLP models. Indeed, tokenization involves splitting the text into smaller units, but the latter are not necessarily words. For example, in our case, the tokenizer will split the text into sub-words.

## Data preparation

We have the model ✅ and the tokenizer ✅, and we must prepare the training and validation datasets.
To do so, we split the dataset into two parts: a training set containing 80% of the original corpus and a validation set containing the remaining 20%.

We also use the `DatasetDict` class from `datasets` package to store the training and validation sets. This class provides many methods to manipulate the data efficiently. For example, we can run a pre-processing step to pre-tokenize the text and avoid running the tokenizer during training.

The tokenizer maps each token to an index in the vocabulary creating the `input_ids` vector. The expected output is a vector of the same length as `input_ids` but containing the index of the target tokens.

**Masked Language Modeling (MLM)**

Masked language modeling (MLM) is the task of randomly masking some words in the input and asking the model to guess the original word. It is a *self-supervised* objective that one can use to train the model without any labeled data. Indeed, the expected output for each masked word is simply the index of the original word. Let's see a simple example of MLM below.

![image](https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/MLM.png)

For training the model, we chose to mask 15% of the tokens in the training set. Given a sentence, we randomly decide to mask a token, and we replace it with the special token `[MASK]`. The model is then trained to predict the original token. 

Using the MLM objective, **we use as labels the original token ids**. During the tokenization step, we set the expected output (e.g., `labels` vector) as the original token ids (`input_ids`). During training, we will randomly mask some tokens and let the model try to predict the original token ids.
The *collate* function (`collate_fn`) will be responsible for this masking step.

🚨 Given the computational resources required for running the pre-training, we only sample 5% of the TatoEBA collection.

```python
DATASET_SAMPLE = 0.05 

data = df["sent_en"].drop_duplicates()

# sample to ease compute
data = data.sample(frac=DATASET_SAMPLE, random_state=42)

train_df, val_df = train_test_split(data, train_size=0.8, random_state=42)
print("Train", train_df.shape, "Valid", val_df.shape)

raw_datasets = DatasetDict(
    {
        "train": Dataset.from_dict({"text": train_df.tolist()}),
        "valid": Dataset.from_dict({"text": val_df.tolist()}),
    }
)

def preprocess(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    This function tokenizes the input sentences and adds the special tokens.
    :param examples: The input sentences.
    :return: The tokenized sentences.
    """
    out = tokenizer.encode_batch(examples["text"])
    return {
        "input_ids": [o.ids for o in out],
        "attention_mask": [o.attention_mask for o in out],
        "special_tokens_mask": [o.special_tokens_mask for o in out],
        # "labels": [o.ids for o in out], # we don't need labels!
    }


proc_datasets = raw_datasets.map(
    preprocess, batched=True, batch_size=4000, remove_columns=["text"]
)
proc_datasets["train"]
```

```python
def collate_fn(batch):
    """
    Collate function that prepares the input for the MLM language modeling task.
    The input tokens are masked according to the MASK_PROBABILITY to generate the 'labels'.

    EXERCISE
    """
    input_ids = jnp.array([s["input_ids"] for s in batch])
    attention_mask = jnp.array([s["attention_mask"] for s in batch])
    special_tokens_mask = jnp.array([s["special_tokens_mask"] for s in batch])
    labels = input_ids.copy()

    special_tokens_mask = special_tokens_mask.astype("bool")
    masked_indices = jax.random.bernoulli(
        next(rng_iter), MASK_PROBABILITY, labels.shape
    ).astype("bool")
    masked_indices = jnp.where(special_tokens_mask, False, masked_indices)

    # Set labels to -100 for non-[MASK] tokens (we will use this while defining the loss function)
    labels = jnp.where(~masked_indices, -100, labels)

    input_ids = jnp.where(masked_indices, tokenizer.token_to_id("[MASK]"), input_ids)

    item = {
        "input_ids": input_ids,
        "attention_mask": jnp.expand_dims(
            attention_mask, 1
        ),  # attention mask must be broadcastable to (B,...,S,S)!
        "labels": labels,
    }
    return item


train_loader = DataLoader(
    proc_datasets["train"], batch_size=BATCH_SIZE, collate_fn=collate_fn
)
valid_loader = DataLoader(
    proc_datasets["valid"], batch_size=BATCH_SIZE, collate_fn=collate_fn
)
```

In the last cell, we used `torch.utils.data.DataLoader`. A **dataloader** is a container that provides an iterable interface over a dataset. It handles the batching and shuffling and is useful for providing data to the training and validation loops. It also provides a specific parameter to use a `collate_fn` which is the function that handles the creation of the batches. In our example, this is where we randomly mask some tokens for the MLM objective.

## Defining a Language Model (with a JAX/Haiku Transform)

At this point, we have the model ✅, the tokenizer ✅, and the data for training and validation ✅. The next step is to define the training loop and all the steps that need to be done inside it.

Similarly to each component of the mode, we will implement the training loop using [Haiku](https://github.com/deepmind/dm-haiku). Before implementing our model let's first recall a very important concept in JAX/Haiku: **the model must be a pure function**. This means that it cannot access any data that is not passed to it. This is a very powerful concept because it makes it really easy to parallelize your model and it allows for automatic differentiation 💪.

Thanks to the `hk.transform` module, we can define a function `mlm_language_model` that takes as input the `input_ids` and the `mask` and runs the model. It also takes as input a flag `is_train` that indicates whether we are training or evaluating the model. This is important because we need to know when to use the `dropout` operations (i.e., only during training).

```python
@hk.transform
def mlm_language_model(input_ids, mask, is_train=True):
    """
    MLM language model as an haiku pure transformation.
    :param input_ids: The input token ids.
    :param mask: The attention mask.
    :param is_train: Whether the model is in training mode.
    :return: The logits corresponding to the output of the model.
    """
    
    """
    EXERCISE
    """
    pe = PositionalEncoding(D_MODEL, MAX_SEQ_LEN, P_DROPOUT)
    embeddings = Embeddings(D_MODEL, VOCAB_SIZE)
    encoder = TransformerEncoder(NUM_LAYERS, NUM_HEADS, D_MODEL, D_FF, P_DROPOUT)

    # get input token embeddings
    input_embs = embeddings(input_ids)
    if len(input_embs.shape) == 2:
        input_embs = jnp.expand_dims(input_embs, 0)  # (1,MAX_SEQ_LEN,D_MODEL)

    # sum positional encodings
    input_embs = pe(input_embs, is_train=is_train)  # (B,MAX_SEQ_LEN,d_model)

    # encode using the transformer encoder stack
    output_embs = encoder(input_embs, mask=mask, is_train=is_train)

    # decode each position into a probability distribution over vocabulary tokens
    out = hk.Linear(D_MODEL, name="dec_lin_1")(output_embs)
    out = jax.nn.relu(out)
    out = hk.LayerNorm(
        axis=-1, param_axis=-1, create_scale=True, create_offset=True, name="dec_norm"
    )(out)
    out = hk.Linear(VOCAB_SIZE, name="dec_lin_2")(out)  # logits
    return out
```

```python
# testing the LM
input_ids = jnp.array(tokenizer.encode("Hello my friend").ids) # encode a sentence
rng_key = next(rng_iter) # get a new random key
mask = jax.random.randint(rng, (1, 1, input_ids.shape[-1]), minval=0, maxval=2) # create a mask
params = mlm_language_model.init(rng_key, input_ids, None, True) # initialize the model
out = mlm_language_model.apply(
    params=params, rng=rng_key, input_ids=input_ids, mask=None, is_train=True
) # apply the model to the input sentence encoded at the previous step
print(out.shape)  # output should be of shape (1,MAX_SEQ_LEN,VOCAB_SIZE)
```

## Training accessories 💍

Before writing the training loop, we need to define some accessories used during the training. These accessories include the **training state** (e.g., the mode parameters and the optimizer state), the **loss function**, and the **train and evaluation steps**.

**Training state**

The training state will allow us to keep track of the training progress and contains all the information we need, e.g., the model parameters and the optimizer. Implementing the model using JAX makes it easy to define a training state.

```python
class TrainingState(NamedTuple):
    """
    The training state is a named tuple containing the model parameters and the optimizer state.
    """
    params: hk.Params # model parameters
    opt_state: optax.OptState # optimizer state
```

Before running the actual training, we need to initialize the network (you have already seen this when testing the previous modules) and an optimizer. 

We will use the `Adam` optimizer, which is a gradient-based optimization algorithm that adapts the learning rate based on the estimated first and second moments of the gradients. It is a very popular optimization algorithm and has shown great results in practice.

**Resources**
- Adam optimizer: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

```python
# Initialise network and optimiser; note we draw an input to get shapes.
sample = proc_datasets["train"][0]
input_ids, attention_mask = map(
    jnp.array, (sample["input_ids"], sample["attention_mask"])
)
rng_key = next(rng_iter)
init_params = mlm_language_model.init(rng_key, input_ids, attention_mask, True)

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP_VALUE),
    optax.adam(LEARNING_RATE),
)
init_opt_state = optimizer.init(init_params)

# initialize the training state class
state = TrainingState(init_params, init_opt_state)
```


**Loss Function**

The loss function is the objective that we want to minimize during training. In general, the loss function needs to be differentiable to compute the gradient of the error using automatic differentiation. In our case, we will use the *Cross Entropy* loss traditionally used for classification tasks. The `optax` library has a function that allows us to easily define the loss function ([see the docs here](https://optax.readthedocs.io/en/latest/api.html#optax.softmax_cross_entropy_with_integer_labels)).

🚨🚨🚨

While implementing the loss function, make sure to carefully manage *padding*. You may not want to consider the padding positions when calculating the loss function. Thus, the loss function should only consider the valid positions.

```python
def loss_fn(params: hk.Params, batch, rng) -> jnp.ndarray:
    """
    The loss function for the MLM language modeling task.
    It computes the cross entropy loss between the logits and the labels.

    :param params: The model parameters.
    :param batch: The batch of data.
    :param rng: The random number generator.
    :return: The value of the loss computed on the batch.
    """
    logits = mlm_language_model.apply(
        params=params,
        rng=rng,
        input_ids=batch["input_ids"],
        mask=batch["attention_mask"],
        is_train=True,
    )    
    
    label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
    # if the number is negative, jax.nn.one_hot() return a jnp.zeros(VOCAB_SIZE)
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch["labels"], VOCAB_SIZE)) * label_mask
    loss = jnp.where(jnp.isnan(loss), 0, loss)
    
    # take average
    loss = loss.sum() / label_mask.sum()
    return loss
```

**Training and Evaluation steps**

The training and evaluation steps are the core of the training loop. They implement the training loop logic.

**Training step**: For each batch, it should (i) forward propagate the batch through the model, (ii) compute the loss and the gradient and then (iii) update the model parameters using the optimizer.

**Evaluation step**: For each batch, it should (i) forward propagate the batch through the model and then (ii) compute and return the loss that corresponds to the current model parameters.

```python
@jax.jit
def train_step(state, batch, rng_key) -> TrainingState:
    """
    The training step function. It computes the loss and gradients, and updates the model parameters.
    
    :param state: The training state.
    :param batch: The batch of data.
    :param rng_key: The key for the random number generator.
    :return: The updated training state, the metrics (training loss) and the random number generator key.
    """
    rng_key, rng = jax.random.split(rng_key)

    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(state.params, batch, rng_key)

    updates, opt_state = optimizer.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(params, opt_state)
    metrics = {"train_loss": loss}

    return new_state, metrics, rng_key
```

```python
@jax.jit
def eval_step(params: hk.Params, batch) -> jnp.ndarray:
    """
    The evaluation step function. It computes the loss on the batch.
    
    :param params: The model parameters.
    :param batch: The batch of data.
    :return: The value of the loss computed on the batch.

    """
    logits = hk.without_apply_rng(mlm_language_model).apply(
        params=params,
        input_ids=batch["input_ids"],
        mask=batch["attention_mask"],
        is_train=False,
    )

    label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
    # if the number is negative, jax.nn.one_hot() return a jnp.zeros(VOCAB_SIZE)
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch["labels"], VOCAB_SIZE)) * label_mask
    loss = jnp.where(jnp.isnan(loss), 0, loss)
    # take average
    loss = loss.sum() / label_mask.sum()
    return loss
```


## The Training Loop

The training loop will execute the training and evaluation steps by iterating over the training and validation datasets. It relies on hyperparameters such as the number of epochs `EPOCHS` and the number of steps between each evaluation `EVAL_STEPS` (you typically do not want to wait until the end of the epoch to assess your model, nor do it so often that the training slows down).

**Checkpointing**

The training loop also includes the checkpointing logic, which saves the model parameters to disk at each evaluation step if the loss on the evaluation has improved. 

**Debugging**

Unfortunately, debugging JIT-ed code (as the one we are using within our training loop) can be pretty tricky. It is because JAX compiles the functions before executing them, so it is impossible to set breakpoints or print traces.
If you want to set checkpoints or print variables, you can comment out `@jax.jit` from either your `train_step` or `eval_step` definitions.

Read [here](https://github.com/google/jax/issues/196) why you cannot print in JIT-compiled functions.

**Experiment tracking**

Tracking is your training dynamics if fundamental to inspect if any bug occurs or everything proceeds as expected. Today, many tracking tools expose handy API to streamline experiment tracking. Today, we will use Tensorboard, which is easy to integrate into Jupyter Lab / Google Colab. 

First, we set a `LOG_STEPS` variable responsible for tracking the training loss for each fixed number of steps. Then, we use a `SummaryWriter` object to log metrics every `LOG_STEPS`. Finally, we can observe our logged metrics by opening a dedicated tab within a notebook cell: execute the following cell to load the tensorboard extension (if you are running the notebook locally, you have to install tensorboard beforehand) and open it.

```python
# The training loop
# It is a simple for loop that iterates over the training set and evaluates on the validation set.

# The hyperparameters used for training and evaluation
EPOCHS = 30  # @param {type:"number"}
EVAL_STEPS = 500  # @param {type:"number"}
MAX_STEPS = 200  # @param {type:"number"}
LOG_STEPS = 200

writer = SummaryWriter()
pbar = tqdm(desc="Train step", total=EPOCHS * len(train_loader))
step = 0
loop_metrics = {"train_loss": None, "eval_loss": None}
best_eval_loss = float("inf")

for epoch in range(EPOCHS):

    for batch in train_loader:

        state, metrics, rng_key = train_step(state, batch, rng_key)
        loop_metrics.update(metrics)
        pbar.update(1)
        step += 1

        # Evaluation loop, no optimization is involved here.
        if step % EVAL_STEPS == 0:
            ebar = tqdm(desc="Eval step", total=len(valid_loader), leave=False)

            losses = list()
            for batch in valid_loader:
                loss = eval_step(state.params, batch)
                losses.append(loss)
                ebar.update(1)
            ebar.close()

            eval_loss = jnp.array(losses).mean()
            loop_metrics["eval_loss"] = eval_loss

            writer.add_scalar("Loss/valid", loop_metrics["eval_loss"].item(), step)

            if eval_loss.item() < best_eval_loss:
                best_eval_loss = eval_loss.item()
                # Save the params training state (and params) to disk
                with open(f"ckpt_train_state_{step}.pkl", "wb") as fp:
                    pickle.dump(state, fp)

        if step % LOG_STEPS == 0:
            writer.add_scalar("Loss/train", loop_metrics["train_loss"].item(), step)

        pbar.set_postfix(loop_metrics)

pbar.close()
```

Once concluded the training, we should have a fully functional Language Model!