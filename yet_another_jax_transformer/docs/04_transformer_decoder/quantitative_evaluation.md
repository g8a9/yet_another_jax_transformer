# Quantitative Evaluation with BLEU

Well done! By now, you should have a trained full encoder-decoder Transformer capable of translating English to Italian. But how good is it?
The evaluation of machine translation systems encompasses several aspects, and practitioners can look at different criteria.

In this tutorial, we will first assess the **translation quality**. Several metrics measure how *close* is the automatically generated sentence to a given gold human translation.
BLUE is an established metric to score a translated candidate sentence against one or more reference texts. 
BLUE is based on n-gram precision between the candidate and all the references text plus a regularization factor (see the second resource for a basic explanation). 

The metric ranges in \[0,1\], and higher scores are best.  

**Resources**
- Original paper: [Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- Introduction to BLEU scores: [Professor Christopher Potts @ Stanford CS224U](https://youtu.be/l-DERqIJjCY?t=362) 

The BLEU implementation is present in many NLP toolkits; we will use `evaluate` here. The following cells show a simple computation over the translation generated with our best checkpoint.

```python
bleu = evaluate.load("bleu")

src = "today we are talking about peace."
gold = ["oggi parliamo di pace.", "parleremo di pace oggi.", "oggi, parleremo di pace."]
translation = translate(state.params, src, tokenizer)
print("Translation:", translation)
bleu.compute(references=[gold], predictions=[translation])
```

## Evaluation on the Europarl held-out set

Let's now translate the Europarl testing set and evaluate our system on BLEU.

Remember that in a standard Colab instance, that will require ~30 minutes (training will look slow at the beginning, but it speeds up soon).

```python
europarl_test = load_dataset("g8a9/europarl_en-it", split="test")

def translate_texts(params, tokenizer, texts):
    """Translate a corpus."""
    return [translate(params, q, tokenizer) for q in tqdm(texts)]

translations = translate_texts(state.params, tokenizer, europarl_test["sent_en"])
```

If you do not want to wait, download the translations we pre-computed with the checkpoint linked above running the cell below.

```bash
curl -LO https://huggingface.co/morenolq/m2l_2022_nlp/resolve/main/europarl_test_translated.txt
```

```python
with open("europarl_test_translated.txt") as fp:
    translations = [l.strip() for l in fp.readlines()]
```

Finally, let's score against our human annotated translations.

```python
bleu.compute(references=europarl_test["sent_it"], predictions=translations)
```

## Evaluation on the Tatoeba held-out test set

Note that depending on the model and checkpoint you are using, these texts can be either in-context (if you are using a Tatoeba model) or out-of-context (if you are using a Europarl model).

```python
translations = translate_texts(state.params, tokenizer, test_df_mt["sent_en"])
bleu.compute(references=test_df_mt["sent_it"].tolist(), predictions=translations)
```