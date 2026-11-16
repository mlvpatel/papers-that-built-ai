# 02 Word2Vec: words as coordinates

Paper: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), Mikolov, Chen, Corrado, Dean (Google, 2013). [PDF](https://arxiv.org/pdf/1301.3781)

## What was broken

Step 01's networks learn from numbers, but most language systems treated each word as an index in a vocabulary, with no notion that some words are similar (page 1). Neural language models could learn word vectors, but they were slow: none had been trained on more than a few hundred million words (pages 1 and 2).

## The idea

Train a very small model with one job: from each word, predict the words around it. This is the skip-gram model (Figure 1, page 5). Words that appear in similar places end up with similar vectors. With no hidden layer the models are cheap: the paper reports high-quality vectors from 1.6 billion words in less than a day (page 1), and skip-gram, the slower of its two models, took about two days on that data (Table 5, page 8).

The paper made the output layer cheap with a hierarchical softmax (page 3). The demo uses negative sampling instead, from the same authors' follow-up paper later in 2013 (reference 21, page 12). The demo scores the real neighbour high and five random words low.

```python
pos_loss = F.logsigmoid((u * vout[o]).sum(1))
neg_loss = F.logsigmoid(-(vout[neg] @ u.unsqueeze(2)).squeeze(2)).sum(1)
```

## Run it

```bash
python papers/02-word2vec/demo.py
```

The demo trains 64-dimensional vectors on the dialogue of Tiny Shakespeare. The score is the average similarity of 16 related pairs (king/queen, day/night, father/son and others, fixed before the first run) minus the average similarity of 1,000 random pairs. Zero means the related pairs are no closer than chance.

Measured on an Apple M5 Pro (48 GB) GPU, 16 November 2026:

```text
194,240 words, 3,258 in the vocabulary, 449,010 training pairs, 16 related pairs scored
  step     0  loss 4.159
  step  1000  loss 2.427
  step  2000  loss 2.182
  step  3000  loss 2.111
  step  4000  loss 2.087
  step  4999  loss 2.060
  king  untrained: share, touching, tower, awhile, commands
        trained:   henry, warwick, spite, seated, warwick's
  love  untrained: pisa, cabin, lamb, cost, censure
        trained:   tender, 'hic, lute, livest, princess
 night  untrained: cousin, court, moving, spread, renowned
        trained:   dim, bed, torch, morrow, midnight
RESULT before=-0.016 after=0.258 seconds=12 device=mps
```

Untrained, the related pairs score no better than random ones. After 12 seconds of training they score 0.258 higher. *Night* now sits next to dim, bed, torch, morrow and midnight, and *king* next to Henry and Warwick, names that fill the history plays in this text. *Love* is noisier: 200,000 words is a very small corpus next to the paper's billions.

On a CPU, use `--quick` (1,000 steps). It reaches the same score in 3 seconds, with slightly different neighbours (*night*: midnight, day, nights):

```text
RESULT before=-0.016 after=0.259 seconds=3 device=cpu
```

## The paper's numbers

The paper tests vectors with analogy questions, such as finding the word that is to *small* as *biggest* is to *big* by vector arithmetic (page 5). Its test set has 8,869 semantic and 10,675 syntactic questions (page 6). The best-known example: vector("King") − vector("Man") + vector("Woman") is closest to vector("Queen") (page 2).

Accuracy with 640-dimensional vectors, all models trained on the same data (Table 3, page 7):

| Model | Semantic questions | Syntactic questions |
|---|---|---|
| Recurrent language model | 9% | 36% |
| Feedforward language model | 23% | 53% |
| CBOW (this paper) | 24% | 64% |
| Skip-gram (this paper) | 55% | 59% |

## What broke next

A word vector holds a word's meaning, but a model still has to read a whole sentence and keep it in mind. Translation models of the time squeezed each sentence into a single fixed-length vector before translating it, the bottleneck the [step 03](../03-attention/) paper set out to remove in 2014 with attention, which lets the model look back at every word.
