# 5️⃣ Transformer and Neural Machine Translation 

Until now, we focused entirely on the Encoder part of the architecture. As we have seen, one can build a Language Model that can work as a generic "encoder" of tokens and then fine-tune it.

However, as we examined in Section 1 of the tutorial, the Transformer architecture also has a **Decoder** part, i.e., a neural network that can "decode" some sequence into something else.\*

Indeed, the original application of the Transformer was Neural Machine Translation (NMT, or MT). In this task, the network is presented with samples composed of a *source sentence* in a given language, say English, and a *target sentence* in a different language, say Italian, as the result of the translation.

In this part of the tutorial, we will conclude the Transformer implementation by sketching the Transformer Decoder. Then, we will test the complete architecture in the task of MT. Specifically, we will train an MT system from scratch on the TatoEBA dataset. 

Before we start coding, let's review how the Transformer learns to map a source sequence into a target sequence.


\* *There exist some language models trained to predict the next word in a sentence. These models are "decoder-only" as their only purpose is to "decode" a given context (the words already seen) into the subsequent token. You might have heard about some of them: GPT-\*, OPT, or BLOOM.*

## Mapping Sources ➡️ and Targets ⬅️

But how does the Transformer work? How do we train it in a sequence-to-sequence setup, and how do we use it to transform a source into a target sequence? Let's cover these questions before turning to the actual code.

**How does it work?**

The Transformer processes a source sequence producing a *contextualized, dense vector representation* of each token of the sequence. This step is what you implemented in Sections 2️⃣ and 3️⃣ with the Encoder.

At the same time, the model processes the target sequence with the Decoder. However, the Decoder does not work in isolation: we inject the source sequence and let the model learn a mapping between source and target. This operation is what you will implement in the remainder of this section.

**How do I train it?**

We provide the model with both the source and target sequences at training time. The Encoder contextualizes the source tokens, and the Decoder **distills** this information in the target sequence representations, using a **cross-attention layer**. Additionally, each token in the Decoder is processed in an **autoregressive/masked** fashion, meaning that it expresses an attention weight only to previous/past tokens.

☝️ these *source distillation* and *autoregressive/masked attention* features are the two crucial differences between the Encoder and Decoder.


**How do I use it to go from source to target?**

Once it is trained, you have to "decode" the source sentence. You typically do that by:
1. encoding the source, as usual;
2. produce one token at a time. Each token will run self-attention on the past decoded tokens and cross-attention on the source sentence.

See the animation below for a graphical representation of decoding.

![mt_example](https://github.com/g8a9/graphics/blob/main/mt_example.gif?raw=true)
