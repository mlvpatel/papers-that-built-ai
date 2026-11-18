# 03 Attention: look back at every input word

Paper: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Bahdanau, Cho, Bengio (2014). [PDF](https://arxiv.org/pdf/1409.0473)

## What was broken

Neural translation models of 2014 were encoder-decoders: one recurrent network read the source sentence into a single fixed-length vector, and another wrote the translation from that vector. The paper's authors conjectured that this fixed-length vector is "a bottleneck" (page 1). The longer the sentence, the more has to fit into the same vector.

## The idea

Keep every encoder state instead of only the last one. At each output step, the decoder scores every input position, turns the scores into weights, and reads a weighted mix of the encoder states. The weights are learned along with everything else, so the model learns where to look. The paper notes that its model turns back into the plain encoder-decoder if the context is fixed to the encoder's final state (page 14), and that is exactly the demo's model without attention:

```python
if self.attention:  # Bahdanau: score every input position, then mix them
    scores = self.v(torch.tanh(keys + self.w_dec(h).unsqueeze(1))).squeeze(2)
    context = (scores.softmax(1).unsqueeze(2) * enc).sum(1)
else:  # the whole input squeezed into the encoder's last state
    context = enc[:, -1]
```

## Run it

```bash
python papers/03-attention/demo.py
```

Run time on an Apple M5 Pro: 63 s for a full run on the GPU, 17 s with `--quick` on the CPU.

Two recurrent encoder-decoders, the same network from the same seed, learn to reverse random strings of 4 to 32 letters. Then each reverses 500 new strings at lengths 8, 16 and 32. The score is the share of output characters that are correct.

Measured on an Apple M5 Pro (48 GB) GPU, 16 November 2026:

```text
  training without attention
    step    0  loss 2.307
    step  500  loss 0.932
    step 1000  loss 1.135
    step 1500  loss 1.261
    step 1999  loss 1.360
  training with attention
    step    0  loss 2.305
    step  500  loss 0.160
    step 1000  loss 0.104
    step 1500  loss 0.059
    step 1999  loss 0.197
length | without attention | with attention   (characters correct)
     8 |             99.4% |          99.9%
    16 |             58.4% |          98.7%
    32 |             26.0% |          94.2%
RESULT before=26.0 after=94.2 seconds=63 device=mps
```

Both models reverse 8-letter strings almost perfectly. Without attention, accuracy falls to 58.4% at 16 letters and 26.0% at 32, close to the 10% of random guessing. With attention it stays at 94.2%. GPU runs vary a little from run to run: an earlier run on 15 November gave 24.5% and 96.7% at length 32. (The loss lines are single batches of random length, so they jump around.)

On a CPU, use `--quick` (400 steps each). The same pattern, less trained:

```text
RESULT before=23.4 after=65.3 seconds=17 device=cpu
```

The first full run of this demo had no gradient clipping. The attention model's loss fell to 0.199, climbed back to 1.989 by step 2,000, and it finished at 24.2% at length 32, no better than the model without attention. The paper caps the size (norm) of each gradient at 1 (pages 14 and 15), and the demo now does the same; the results above are from that version.

## The paper's numbers

English-to-French translation quality (BLEU, higher is better), with models trained on sentences of up to 30 or 50 words (Table 1, page 7):

| Model | All test sentences | Sentences with no unknown words |
|---|---|---|
| Encoder-decoder, up to 30 words | 13.93 | 24.19 |
| With attention, up to 30 words | 21.50 | 31.44 |
| Encoder-decoder, up to 50 words | 17.82 | 26.71 |
| With attention, up to 50 words | 26.75 | 34.16 |
| Moses, a phrase-based system | 33.30 | 35.63 |

By sentence length, the encoder-decoder's quality "dramatically drops as the length of the sentences increases", while the attention model trained on 50-word sentences "shows no performance deterioration even with sentences of length 50 or more" (Figure 2, page 5; quoted text on page 6).

## What broke next

Attention fixed the bottleneck, but it still sat on top of a recurrent network that reads one word at a time. The Transformer paper names the cost: that "inherently sequential nature precludes parallelization within training examples" (Vaswani et al., page 2). [Step 04](../04-transformer/) keeps the attention and removes the recurrence.
