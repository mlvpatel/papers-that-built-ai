# 04 Transformer: attention instead of recurrence

Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google, 2017). [PDF](https://arxiv.org/pdf/1706.03762)

## What was broken

The best language and translation models of 2017 were recurrent networks, which read a sentence one position at a time and carry a hidden state forward. The paper names the cost: this "inherently sequential nature precludes parallelization within training examples" (page 2). A recurrent layer needs as many sequential steps as there are positions, so long texts train slowly however many GPUs you have.

## The idea

Drop recurrence and let every position look at every other position directly through attention. A self-attention layer connects all positions in a constant number of sequential steps, where a recurrent layer needs one step per position (Table 1, page 6). All positions are computed at once, which suits GPUs.

In the demo, the recurrent model reads one character after another, and the Transformer block attends to all earlier characters in one call (shown simplified; the demo reshapes the heads around it):

```python
out, _ = self.gru(self.tok(x))  # one character after another
```

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # all positions at once
```

## Run it

```bash
python papers/04-transformer/demo.py
```

Two character models of about the same size, trained on Tiny Shakespeare with the same data, seed and 1,500 steps: a GRU (a recurrent network) and a 2-block Transformer. Lower validation loss is better.

Measured on an Apple M5 Pro (48 GB) GPU, 15 November 2026:

```text
  GRU         step    0  loss 4.188
  GRU         step  500  loss 1.622
  GRU         step 1000  loss 1.459
GRU         428,097 parameters, val loss 1.605, trained in 40 s
  Transformer step    0  loss 4.341
  Transformer step  500  loss 1.851
  Transformer step 1000  loss 1.685
Transformer 421,697 parameters, val loss 1.724, trained in 17 s
RESULT before=1.61 after=1.72 seconds=59 device=mps gru_seconds=40 transformer_seconds=17
```

The two results point in different directions, and both are real. Step for step, the GRU learns slightly better here: 1.61 against 1.72. Each Transformer step is 2.4 times faster, 17 seconds against 40 for the same 1,500 steps. That speed is the paper's argument. At its scale the Transformer won on quality too, but at this scale, with 64-character contexts and 0.4 million parameters, the recurrent model still holds its own per step.

On a CPU, use `--quick` (300 steps each). The same pattern, with a smaller speed gap:

```text
RESULT before=1.85 after=2.12 seconds=19 device=cpu gru_seconds=10 transformer_seconds=7
```

## The paper's numbers

Translation quality (BLEU, higher is better) and training compute on the WMT 2014 test sets (Table 2, page 8):

| Model | English-German | English-French | Training compute, English-German (FLOPs) |
|---|---|---|---|
| GNMT + RL, Google's recurrent system | 24.6 | 39.92 | 2.3 × 10^19 |
| ConvS2S ensemble | 26.36 | 41.29 | 7.7 × 10^19 |
| Transformer, base | 27.3 | 38.1 | 3.3 × 10^18 |
| Transformer, big | 28.4 | 41.8 | 2.3 × 10^19 |

In the paper's words, even the base model "surpasses all previously published models and ensembles, at a fraction of the training cost" (page 8). It trained in 12 hours on eight GPUs (page 7).

## What broke next

A Transformer gets better as you stack more blocks, up to a point. Past that point, a plain deep stack stops learning at all. [Step 05](../05-resnet/) shows the failure and the fix, which the Transformer itself borrowed from ResNet (page 3): every block adds to its input instead of replacing it.
