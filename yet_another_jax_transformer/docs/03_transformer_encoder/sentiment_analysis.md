# Fine-Tuning for Sentiment Classification

Let's focus on a different task for our Transformer Encoder: **Sentiment Analysis**. This task requires determining if the sentiment of a given piece of text is positive or negative.

Remember, the pre-training implemented before is not specific to any downstream task: we were training our Transformer on a broad set of input data and learning representations that can be useful when *adapted* to different tasks. Indeed, we will now build a classifier based on our pre-trained Transformer to perform sentiment analysis.

In practice, we will train a language model with a sentiment classification *head*: instead of producing a probability distribution across all the words in our vocabulary, we will produce a class probability over a set of labels.

We will use the Stanford Sentiment Treebank V2 dataset (SST-2). The dataset contains 11,855 sentences extracted from movie reviews. The sentences have been labeled with positive (1) or negative (0) sentiments.

📚 **Resources**

- SST-2 paper: [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170/)
- SST-2 fields description: [SST-2 on datasets](https://huggingface.co/datasets/sst2)

```python
raw_datasets = load_dataset("glue", "sst2")
raw_datasets
```

## 🧱 Generating a baseline model with TF-IDF and Logistic Regression

We will first build a baseline model to compare with the Transformer model. For this, we will use a TF-IDF representation of the sentences plus a Logistic Regression classifier. The baseline model will be trained using the training set of the SST-2 dataset and evaluated on the evaluation set (unfortunately, the labels of the SST-2 test set dataset are not publicly available).

Run the cell below to train the baseline model.

```python
# TF-IDF baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

MAX_FEATURES = 10_000

X_train, y_train = raw_datasets["train"]["sentence"], raw_datasets["train"]["label"]
X_test, y_test = (
    raw_datasets["validation"]["sentence"],
    raw_datasets["validation"]["label"],
)

tfidf = TfidfVectorizer(max_features=MAX_FEATURES) # instantiate the vectorizer
tfidf = tfidf.fit(X_train + X_test) # fit on all data

X_train = tfidf.transform(X_train) # transform the training data
X_test = tfidf.transform(X_test) # transform the test data

clf = LogisticRegression(random_state=42).fit(X_train, y_train) # train the classifier
y_pred = clf.predict(X_test) # run the classifier on the test data

clf_report = classification_report(y_test, y_pred)
print("\n Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report")
print("======================================================")
print("\n", clf_report)
```

## Preprocessing the SST-2 dataset

After downloading the dataset, we can process the sentences with our tokenizer. Similarly to the pre-training step, we preprocess all sentences offline and avoid doing it at each training iteration.

Once data are processed, we can create the data loaders that will feed our training loop.

```python
tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
tokenizer.enable_truncation(True)
tokenizer.enable_padding(length=MAX_SEQ_LEN)

def preprocess(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = tokenizer.encode_batch(examples["sentence"])
    return {
        "input_ids": [o.ids for o in out],
        "attention_mask": [o.attention_mask for o in out],
        "special_tokens_mask": [o.special_tokens_mask for o in out],
        "labels": examples["label"],
    }


proc_datasets = raw_datasets.map(
    preprocess, batched=True, batch_size=4000, remove_columns=["sentence", "idx"]
)

def collate_fn(batch):
    """
    Collate function to generate the input for the model and their corresponding labels.
    In this case, the labels corresponds to the expected class of each input sequence.
    """
    item = {
        "input_ids": jnp.array([s["input_ids"] for s in batch]),
        "attention_mask": jnp.expand_dims(
            jnp.array([s["attention_mask"] for s in batch]), 1
        ),
        "labels": jnp.array([s["labels"] for s in batch]),
    }
    return item


train_loader = DataLoader(
    proc_datasets["train"], batch_size=BATCH_SIZE, collate_fn=collate_fn
)
valid_loader = DataLoader(
    proc_datasets["validation"], batch_size=BATCH_SIZE, collate_fn=collate_fn
)
test_loader = DataLoader(
    proc_datasets["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn
)
```

## Sentiment Classifier as a Haiku Transform

Similarly to the model trained using MLM, we can write our sentiment classifier as a Haiku Transform.

Our function will contain all the components necessary to perform the classification:
- a pre-trained Transformer Encoder as the base model;
- a Linear layer on top of it. The linear layer maps the Transformer's output representations corresponding to the `[CLS]` token (the first token of the sequence) to two values, the logits of positive and negative sentiment.

We compute Cross Entropy between the prediction and the ground truth labels. Unlike the MLM case, our labels are 0 or 1, so we can use them directly as targets in supervised learning settings.

```python
@hk.transform
def sentiment_classifier(input_ids, mask, is_train=True):
    """
    The sentiment classifier model implemented using Haiku. Each input sequence is
    passed through a transformer encoder and the output is passed through a linear
    layer to obtain the logits.

    :param input_ids: The input sequences.
    :param mask: The attention mask.
    :param is_train: Whether the model is in training mode or not.
    :return: The logits.
    """
    pe = PositionalEncoding(D_MODEL, MAX_SEQ_LEN, P_DROPOUT)
    embeddings = Embeddings(D_MODEL, VOCAB_SIZE)
    encoder = TransformerEncoder(NUM_LAYERS, NUM_HEADS, D_MODEL, D_FF, P_DROPOUT)

    input_embs = embeddings(input_ids)
    if len(input_embs.shape) == 2:
        input_embs = input_embs[None, :, :]
    input_embs = pe(input_embs, is_train=is_train)  # (B,S,d_model)
    output_embs = encoder(input_embs, mask=mask, is_train=is_train)

    # final decoder layer
    out = hk.Linear(D_MODEL)(output_embs)
    out = jax.nn.relu(out)
    out = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)(
        out
    )
    out = out[:, 0, :]  # we use the [CLS] token embedding to represent the sequence and pass it through a linear layer
    out = hk.Linear(2)(out)  # logits
    return out


def loss_fn(params: hk.Params, batch, rng) -> jnp.ndarray:
    """
    The loss function for the model. It takes the model parameters, the input batch
    and the random number generator as input and returns the loss.

    :param params: The model parameters.
    :param batch: The input batch.
    :param rng: The random number generator.
    :return: The loss.
    """
    logits = sentiment_classifier.apply(
        params=params,
        rng=rng,
        input_ids=batch["input_ids"],
        mask=batch["attention_mask"],
        is_train=True,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
    return loss.mean()


@jax.jit
def deterministic_forward(params: hk.Params, batch) -> jnp.ndarray:
    """
    This function is used to forward the model in a deterministic way. 
    It uses without_apply_rng to disable the use of the random number generator.
    It takes the model parameters and the input batch as input and returns the logits.

    :param params: The model parameters.
    :param batch: The input batch.
    :return: The logits.
    """
    return hk.without_apply_rng(sentiment_classifier).apply(
        params=params,
        input_ids=batch["input_ids"],
        mask=batch["attention_mask"],
        is_train=False,
    )


def eval_step(params: hk.Params, batch) -> jnp.ndarray:
    """Evaluation step."""
    logits = deterministic_forward(params, batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
    acc = (logits.argmax(-1) == batch["labels"]).sum() / batch["labels"].shape[0]
    return loss.mean(), acc
```

## Training loop

The training loop is similar to the MLM case. However, we do not need to mask the input to the Transformer since we are not training for a language modeling task.

Given the supervised settings, we can evaluate the model against the validation set during training. We will monitor the validation accuracy, i.e., the percentage of correctly classified samples. The training loop tracks both the loss and the validation accuracy so that we can observe their dynamics during training.

```python
# Initialise network and optimiser; note we draw an input to get shapes.
sample = proc_datasets["train"][0]
input_ids, attention_mask = map(
    jnp.array, (sample["input_ids"], sample["attention_mask"])
)
rng = next(rng_iter)
init_params = sentiment_classifier.init(rng, input_ids, attention_mask, True)

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP_VALUE),
    optax.adam(learning_rate=5e-5),
)
init_opt_state = optimizer.init(init_params)

# initialize the training state class
state = TrainingState(init_params, init_opt_state)
```

Finally, we are ready to run the training.

```python
# Training & evaluation loop.

EPOCHS = 10
EVAL_STEPS = 500
LOG_STEPS = 100

writer = SummaryWriter()
pbar = tqdm(desc="Train step", total=EPOCHS * len(train_loader))
step = 0
loop_metrics = {"train_loss": None, "eval_loss": None}
best_eval_loss = float("inf")
best_eval_acc = float("-inf")

for epoch in range(EPOCHS):

    for batch in train_loader:
        # Do SGD on a batch of training examples.
        state, metrics, rng_key = train_step(state, batch, rng_key)
        loop_metrics.update(metrics)
        pbar.update(1)
        step += 1

        if step % EVAL_STEPS == 0:
            metrics = list()
            for batch in tqdm(
                valid_loader, desc="Eval", total=len(valid_loader), leave=False
            ):
                metrics.append(eval_step(state.params, batch))

            eval_loss = jnp.array([m[0] for m in metrics]).mean()
            eval_acc = jnp.array([m[1] for m in metrics]).mean()
            loop_metrics["eval_loss"] = eval_loss
            loop_metrics["eval_acc"] = eval_acc

            writer.add_scalar("Loss/valid", loop_metrics["eval_loss"].item(), step)
            writer.add_scalar("Acc/valid", loop_metrics["eval_acc"].item(), step)

            if eval_acc.item() > best_eval_acc:
                best_eval_loss = eval_loss.item()
                best_eval_acc = eval_acc.item()
                best_eval_ckpt = f"sentiment_class_state_{step}.pkl"

                print(best_eval_acc, best_eval_ckpt)
                # Save the params training state (and params) to disk
                with open(best_eval_ckpt, "wb") as fp:
                    pickle.dump(state, fp)

        if step % LOG_STEPS == 0:
            writer.add_scalar("Loss/train", loop_metrics["train_loss"].item(), step)

        pbar.set_postfix(loop_metrics)
```

## Evaluate the sentiment classification model

After training, we can classify the entire evaluation set using the best checkpoint available and compute the classification report.

Set the variable `model_checkpoint_path` to choose which checkpoint to evaluate.

```python
model_checkpoint_path = "..."
with open(model_checkpoint_path, "rb") as fp:
    state = pickle.load(fp)


def classify(params, tokenizer, batch):
    """Classify a batch of text."""
    logits = deterministic_forward(params, batch) # (B,2)
    return logits.argmax(-1)


y_pred = list()
y_true = list()
for batch in tqdm(valid_loader, desc="Eval", total=len(valid_loader), leave=False):
    out = classify(state.params, tokenizer, batch)
    y_pred.extend(out)
    y_true.extend(batch["labels"])

clf_report = classification_report(y_true, y_pred)
print("\n Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report")
print("======================================================")
print("\n", clf_report)
```

## Randomly initialized 👶 vs. pre-trained 🏋🏼 model

The Transformer encoder trained above is randomly initialized and is trained from scratch. However, the great success of Transformers in many NLP tasks is due to their excellent performance when fine-tuned starting from a pre-trained model. Using a pre-trained model allows us to leverage the large amount of training data used to train the model from scratch for the task we are interested in.

As a result, we can fine-tune the Transformer encoder starting from the pre-trained weights obtained with the MLM objective. The following cell loads the weights of the pre-trained Transformer in the new model we are training for sentiment classification.

Although the model architecture is similar, the sentiment classification model has an additional Linear layer on top of the Transformer, which is randomly initialized. Therefore, we do not (and can not) load the weights of this additional layer into the model.

If you compare the performance of the two models, *which one can reach the highest accuracy?*

You can use both:
- a model that we have already pre-trained for you on large English corpora or,
- the model pre-trained in the previous section


**Use our pre-trained model:** Since everything in Haiku is stateless, you can download the `TrainingState` (which contains the model parameters) and use it in your subsequent `apply` call.
You should also load the pre-trained tokenized paired with the model in this case.

You will need to take care of the hyper-parameters to match the one we used to train it:

```python
NUM_LAYERS = 6
NUM_HEADS = 8
D_MODEL = 128
D_FF = 256
P_DROPOUT = 0.1
MAX_SEQ_LEN = 128
VOCAB_SIZE = 25000
```
Run the cell below to download the files.

**Your own pre-training**: Similarly, you can load the best checkpoint obtained above by loading the checkpoint saved before.

```bash
wget https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/v1_mlm_train_state_362000.pkl
wget https://huggingface.co/morenolq/m2l_2022_nlp/raw/main/v1_en_tokenizer_1M.json
```

```python
# model parameters
NUM_LAYERS = 6
NUM_HEADS = 8
D_MODEL = 128
D_FF = 256
P_DROPOUT = 0.1
MAX_SEQ_LEN = 128
VOCAB_SIZE = 25_000

# Initialise network and optimiser; note we draw an input to get shapes.
sample = proc_datasets["train"][0]
input_ids, attention_mask = map(
    jnp.array, (sample["input_ids"], sample["attention_mask"])
)
rng = next(rng_iter)
init_params = sentiment_classifier.init(rng, input_ids, attention_mask, True)

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP_VALUE),
    optax.adam(learning_rate=5e-5),
)
init_opt_state = optimizer.init(init_params)

# initialize the training state class
state = TrainingState(init_params, init_opt_state)

# load the pre-trained model
pretrained_model_path = "v1_mlm_train_state_362000.pkl"

with open(pretrained_model_path, "rb") as fp:
    pretrained_state = pickle.load(fp)

# load the weights from the pre-trained model to the new model
encoder_weights = {
    k: v for k, v in pretrained_state.params.items() if k in state.params
}
print("Found", len(encoder_weights), "pretrained weights")
state.params.update(encoder_weights)

# load pre-trained tokenizer
pretrained_tokenizer_path = "v1_en_tokenizer_1M.json"
tokenizer = tokenizers.Tokenizer.from_file(pretrained_tokenizer_path)
tokenizer.enable_truncation(MAX_SEQ_LEN)
tokenizer.enable_padding(length=MAX_SEQ_LEN)
```

At this point, it is possible to run the training process using the pre-trained weights as a starting point and run the final evaluation to check the model's performance.

```python
EPOCHS = 10
EVAL_STEPS = 500
LOG_STEPS = 100

writer = SummaryWriter()
pbar = tqdm(desc="Train step", total=EPOCHS * len(train_loader))
step = 0
loop_metrics = {"train_loss": None, "eval_loss": None}
best_eval_loss = float("inf")
best_eval_acc = float("-inf")

for epoch in range(EPOCHS):

    for batch in train_loader:
        # Do SGD on a batch of training examples.
        state, metrics, rng_key = train_step(state, batch, rng_key)
        loop_metrics.update(metrics)
        pbar.update(1)
        step += 1

        if step % EVAL_STEPS == 0:
            metrics = list()
            for batch in tqdm(
                valid_loader, desc="Eval", total=len(valid_loader), leave=False
            ):
                metrics.append(eval_step(state.params, batch))

            eval_loss = jnp.array([m[0] for m in metrics]).mean()
            eval_acc = jnp.array([m[1] for m in metrics]).mean()
            loop_metrics["eval_loss"] = eval_loss
            loop_metrics["eval_acc"] = eval_acc

            writer.add_scalar("Loss/valid", loop_metrics["eval_loss"].item(), step)
            writer.add_scalar("Acc/valid", loop_metrics["eval_acc"].item(), step)

            if eval_acc.item() > best_eval_acc:
                best_eval_loss = eval_loss.item()
                best_eval_acc = eval_acc.item()
                best_eval_ckpt = f"ft_sentiment_class_state_{step}.pkl"

                print(best_eval_acc, best_eval_ckpt)
                # Save the params training state (and params) to disk
                with open(best_eval_ckpt, "wb") as fp:
                    pickle.dump(state, fp)

        if step % LOG_STEPS == 0:
            writer.add_scalar("Loss/train", loop_metrics["train_loss"].item(), step)

        pbar.set_postfix(loop_metrics)
```

## Evaluate the sentiment classification model

At the end of the training process, we can run the full evaluation of the best checkpoint available and print the classification report.

```python
model_checkpoint_path = "..."
with open(model_checkpoint_path, "rb") as fp:
    state = pickle.load(fp)


y_pred = list()
y_true = list()
for batch in tqdm(valid_loader, desc="Eval", total=len(valid_loader), leave=False):
    out = classify(state.params, tokenizer, batch)
    y_pred.extend(out)
    y_true.extend(batch["labels"])

clf_report = classification_report(y_true, y_pred)
print("\n Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report - Pre-Training + Fine-Tuning")
print("======================================================")
print("\n", clf_report)
```