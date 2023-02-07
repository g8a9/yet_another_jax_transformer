# (*Bonus*) All the glitter is not gold: Gender Bias in Machine Translation

In this final bonus section, we are introducing the issue of **gender bias** in machine translation systems. 

The section itself does not contain any coding exercise (if you made it so far, it is the least you earned); it is intended to *raise awareness of the issue* and let you reason and discuss mitigation strategies.
To further stress the importance of the matter, please consider that:
- the data we started from, the pipeline we built, the network architecture we chose, and the way we trained it *are not that different* from those used in standard commercial MT systems and leaderboard-breaking academic models;
- the issue of gender bias is compellingly evident even in our simple case study trained on a relatively limited number of samples. 

## The issue

Following related literature, we define "bias," here intended as *social discriminatory bias*, the phenomenon that arises whenever a technology we build actively or passively harms a specific individual, category, or social group. This broad definition has many possible realizations (e.g., what kind of harm is produced, to whom, to what extent -- aspects often loosely defined in NLP works, see Resource 1.): here, we will frame it in the context of machine translation and discrimination against different genders.     

Many factors can cause an MT system to be biased, such as model design choices, data collection, or socio-cultural constructs we collect data from (e.g., consider that women uttered only one-third of the sentences listed in Europarl). We will not dive into details and refer the interested reader to Resource 2.

In this section, we will use an established MT benchmarking dataset, WinoMT, to assess **representational harm**, i.e., whether the model is diminishing the visibility of a group (e.g., representing female references with male ones) or reinforcing social stereotypes (e.g., by associating particular jobs to women) thorough the produced translations. *In the remainder of the section, we will analyze gender as a binary variable (i.e., Male/Female). While we share this over-simplification with most of the relevant research on gender bias in MT, we encourage students to devise solutions to overcome it: what kind of resources or modeling solutions would improve the representation of non-binary individuals?*  

To grasp what we are dealing with, let's pause the theory for a moment and test our model on the sentence: `the scientist was running the experiment.`

```python
print(translate(state.params, "the scientist was running the experiment.", tokenizer))
```

Can you spot anything? (If you are not a native Italian speaker move on to the next section, or find an italian speaker to ask the same question 😉).