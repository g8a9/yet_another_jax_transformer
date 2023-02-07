# Preparation for the MT task

With the Transformer class ready, we can now take our time to go through all the required steps in preparation for the actual training. As you have already seen in Section 3️⃣, we mainly need to:
1. pick a dataset. We need a parallel corpus, where each sample is made of a source and a target sentence;
2. train a tokenizer;
3. preprocess our corpus using the tokenizer.

## Dataset selection

We will use two well-known datasets to train an English-to-Italian translation system.

- [TatoEBA](https://opus.nlpl.eu/Tatoeba.php) is a crowdsourced dataset of sentences annotated on the homonym [website](https://tatoeba.org/en/) by users;
- [Europarl](https://www.statmt.org/europarl/) is a corpus of proceedings of the European Parliament.

We demonstrate the training and provide a few translation examples on TatoEBA since it is smaller and easier to train on. However, you can also download the Europarl dataset by executing the cell below and proceed equivalently (depending on your computing capacity, training on Europarl will be feasible or not).

```python
# Skip this if you are running on Colab or on a low-end GPU or CPU.
raw_datasets = load_dataset("g8a9/europarl_en-it")
```

## Train tokenizer for the Machine Translation task

Here, we opt for training a single tokenizer with double the number of tokens stored compared to the one used for language modeling.

Feel free to test your solution with two different tokenizers (one per language).

```python
# Target tokenizer (SRC+TGT language)
VOCAB_SIZE = 20_000

BATCH_SIZE = 64
NUM_LAYERS = 6
NUM_HEADS = 8
D_MODEL = 128
D_FF = 256
P_DROPOUT = 0.1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-4
GRAD_CLIP_VALUE = 1

# Loading TatoEBA
df = pd.read_csv(
    "it-en.tsv", sep="\t", header=0, names=["id_it", "sent_it", "id_en", "sent_en"]
)

# we will use italian sentences to generate our target tokenizer
it_sentences = df["sent_it"].drop_duplicates().dropna()
en_sentences = df["sent_en"].drop_duplicates().dropna()
print(f"Unique Italian sentences: {len(it_sentences)}")
print("Samples:\n", it_sentences[:5])

# we'll use BPE
mt_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
mt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
mt_tokenizer.normalizer = tokenizers.normalizers.Lowercase()

trainer = tokenizers.trainers.BpeTrainer(
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    vocab_size=VOCAB_SIZE,
    show_progress=True,
    min_frequency=2,
    continuing_subword_prefix="##",
)

mt_tokenizer.train_from_iterator(
    it_sentences.tolist() + en_sentences.tolist(), trainer=trainer
)

bos_id, eos_id = map(mt_tokenizer.token_to_id, ["[BOS]", "[EOS]"])
mt_tokenizer.post_processor = tokenizers.processors.BertProcessing(
    ("[EOS]", eos_id), ("[BOS]", bos_id)
)
mt_tokenizer.enable_truncation(MAX_SEQ_LEN)
mt_tokenizer.enable_padding(length=MAX_SEQ_LEN)

PAD_ID = mt_tokenizer.token_to_id("[PAD]")

mt_tokenizer.save("mt_tokenizer.json")
```

Use the cell below if you want instead to load the tokenizer from disk.

```python
mt_tokenizer = tokenizers.Tokenizer.from_file("mt_tokenizer.json")
mt_tokenizer.enable_truncation(MAX_SEQ_LEN)
mt_tokenizer.enable_padding(length=MAX_SEQ_LEN)
```

## Process and tokenize MT data

As it is not the tutorial's focus, we again provide the code to run the basic preprocessing using `datasets`. Feel free to inspect to understand better every step related to tokenization and data preparation.

```python
DATASET_SAMPLE = 0.1  # @param {type:"number"}

# generate parallel data
mt_df = df.sample(frac=DATASET_SAMPLE, random_state=42)

train_df_mt, test_df_mt = train_test_split(mt_df, test_size=0.2, random_state=42)
val_df_mt, test_df_mt = train_test_split(test_df_mt, test_size=0.5, random_state=42)
print("Train", train_df_mt.shape, "Valid", val_df_mt.shape, "Test", test_df_mt.shape)

raw_datasets = DatasetDict(
    {
        "train": Dataset.from_dict(
            {
                "sent_en": train_df_mt["sent_en"].tolist(),
                "sent_it": train_df_mt["sent_it"].tolist(),
            }
        ),
        "valid": Dataset.from_dict(
            {
                "sent_en": val_df_mt["sent_en"].tolist(),
                "sent_it": val_df_mt["sent_it"].tolist(),
            }
        ),
        "test": Dataset.from_dict(
            {
                "sent_en": test_df_mt["sent_en"].tolist(),
                "sent_it": test_df_mt["sent_it"].tolist(),
            }
        ),
    }
)


def preprocess(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    src = mt_tokenizer.encode_batch(examples["sent_en"], add_special_tokens=False)
    tgt = mt_tokenizer.encode_batch(examples["sent_it"], add_special_tokens=True)

    return {
        "src_ids": [o.ids for o in src],
        "src_mask": [o.attention_mask for o in src],
        "src_special_tokens_mask": [o.special_tokens_mask for o in src],
        "tgt_ids": [o.ids for o in tgt],
    }


proc_datasets = raw_datasets.map(
    preprocess, batched=True, batch_size=4000, remove_columns=["sent_en", "sent_it"]
)

print("First training sample, after processing:", proc_datasets["train"][0])
```

## Utility functions and data structures

Let's define a few functions that will be useful later.

```python
def subsequent_mask(S: int):
    """Mask out subsequent positions.
    
    Given an integer `S`, generate a `1xSxS` matrix containing the attention mask to apply to the sequence.
    The matrix should implement autoregressive attention (left-context attention), i.e., it should mask, for each token at position 'i', every token in [0, 'i'-1).
    
    E.g. 

    MAX_LEN = 8
    SEQ_LEN = 5

    Encoder attention mask:

    [ [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ] ]

    Decoder attention mask:

    [ [1, 0, 0, 0, 0, 0, 0, 0, ]
    [1, 1, 0, 0, 0, 0, 0, 0, ]
    [1, 1, 1, 0, 0, 0, 0, 0, ]
    [1, 1, 1, 1, 0, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 0, 0, 0, ]
    [1, 1, 1, 1, 1, 1, 0, 0, ]
    [1, 1, 1, 1, 1, 1, 1, 0, ]
    [1, 1, 1, 1, 1, 1, 1, 1, ] ]

    """
    attn_shape = (1, S, S)
    subsequent_mask = jnp.triu(jnp.ones(attn_shape), k=1).astype(jnp.uint8)
    return jnp.where(subsequent_mask == 0, 1, 0)

def collate_fn_mt(batch) -> dict:
    """Collate source and target sequences in the batch.

    We also need to define a 'labels' variable.

    You want to produce the following shapes:
    - src: (B,MAX_SEQ_LEN)
    - src_mask: (B,1,MAX_SEQ_LEN)
    - tgt: (B,MAX_SEQ_LEN-1)
    - tgt_mask: (B,MAX_SEQ_LEN-1,MAX_SEQ_LEN-1)
    - labels: (B,MAX_SEQ_LEN-1)
    """
    src = jnp.array([s["src_ids"] for s in batch])
    src_mask = jnp.array([s["src_mask"] for s in batch])
    src_mask = jnp.expand_dims(src_mask, 1)

    tgt_seq = jnp.array([s["tgt_ids"] for s in batch])
    tgt = tgt_seq[:, :-1]  # (B,MAX_SEQ_LEN-1)
    labels = tgt_seq[:, 1:]  # (B,MAX_SEQ_LEN-1)

    tgt_pad = jnp.where(jnp.expand_dims(tgt, axis=1) != PAD_ID, 1, 0)
    tgt_mask = jnp.where(tgt_pad & subsequent_mask(tgt.shape[-1]), 1, 0)

    item = {
        "src": src,
        "src_mask": src_mask,
        "tgt": tgt,
        "tgt_mask": tgt_mask,
        "labels": labels,
    }
    return item


train_loader_mt = DataLoader(
    proc_datasets["train"], batch_size=BATCH_SIZE, collate_fn=collate_fn_mt
)
valid_loader_mt = DataLoader(
    proc_datasets["valid"], batch_size=BATCH_SIZE, collate_fn=collate_fn_mt
)
test_loader_mt = DataLoader(
    proc_datasets["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn_mt
)

print(
    f"Batches Train: {len(train_loader_mt)}",
    f"Valid: {len(valid_loader_mt)}",
    f"Test : {len(test_loader_mt)}",
)
```

