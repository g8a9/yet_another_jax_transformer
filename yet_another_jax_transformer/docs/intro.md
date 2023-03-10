# Yet Another (JAX) Transformer

Following along this book, you will implement every component of the Transformer architecture in [JAX](https://github.com/google/jax). There won't be no tricks or custom things, we will stick to the design as detailed in [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 

```{warning}
The document is currently WIP 🙃
```


## Why YAJT? 

There exist excellent walkthroughs on the Transformer architecture (e.g., [\[1\]](http://nlp.seas.harvard.edu/annotated-transformer/), [\[2\]](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)). YAJT builds on those, but we also tried to

1. improve the educational impact. Along the way, we will touch fundamental ML/NLP topics such as **gradient-based optimization** or **text tokenization** and let you train your brand new transformers to solve **language modeling** and **machine translation**. Moreover, we briefly touch upon social implications and **gender bias**.

2. implement everything, from scratch, and in a low-level JAX.


## Credits

**Authors:** [Giuseppe Attanasio](https://gattanasio.cc/) and [Moreno La Quatra](https://www.mlaquatra.me/).

The content of this book was originally devised as NLP tutorial of the second [Mediterranean Machine Learning Summer School](https://www.m2lschool.org/past-editions/m2l-2022) by the AI Education Foundation.

## Index

```{tableofcontents}
```