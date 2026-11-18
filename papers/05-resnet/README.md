# 05 ResNet: skip connections let a deep model learn

Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He, Zhang, Ren, Sun (Microsoft Research, 2015). [PDF](https://arxiv.org/pdf/1512.03385)

## What was broken

Step 04 built a Transformer. The obvious way to make it smarter is to stack more blocks, and past a certain depth that backfires: the deeper network trains worse, not better. He et al. hit the same wall with image networks in 2015. On CIFAR-10, a plain 56-layer network had higher training error than a 20-layer one (Figure 1, page 1). This is not overfitting, because the training error itself goes up. The paper calls it the degradation problem.

## The idea

Let each block learn a change to its input instead of a whole new output. The block returns `x + F(x)`, so a layer with nothing useful to add can pass `x` through almost untouched. In the demo it is one `x +` per sub-layer, and nothing else changes between the runs:

```python
def forward(self, x):
    if self.skip:  # ResNet: add to x instead of replacing it
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
    x = self.attn(self.ln1(x))
    return self.mlp(self.ln2(x))
```

## Run it

```bash
python papers/05-resnet/demo.py
```

Run time on an Apple M5 Pro: 151 s for a full run on the GPU, 106 s with `--quick` on the CPU.

The demo trains the same character-level Transformer on Tiny Shakespeare three times, from the same seed: 2 blocks without skip connections, 24 blocks without them, and 24 blocks with them. Lower validation loss is better; random guessing scores 4.17.

Measured on an Apple M5 Pro (48 GB) GPU, 16 November 2026, 1,000 steps per run:

```text
1,115,394 characters, 65 distinct; random guessing scores 4.17
shallow  2 blocks, skip=False  step    0  val loss 4.342
shallow  2 blocks, skip=False  step  250  val loss 2.573
shallow  2 blocks, skip=False  step  500  val loss 2.230
shallow  2 blocks, skip=False  step  750  val loss 2.021
shallow  2 blocks, skip=False  final      val loss 1.874
before  24 blocks, skip=False  step    0  val loss 4.307
before  24 blocks, skip=False  step  250  val loss 3.353
before  24 blocks, skip=False  step  500  val loss 3.333
before  24 blocks, skip=False  step  750  val loss 3.350
before  24 blocks, skip=False  final      val loss 3.360
after   24 blocks, skip=True   step    0  val loss 4.414
after   24 blocks, skip=True   step  250  val loss 2.082
after   24 blocks, skip=True   step  500  val loss 1.840
after   24 blocks, skip=True   step  750  val loss 1.719
after   24 blocks, skip=True   final      val loss 1.671
RESULT before=3.36 after=1.67 seconds=151 device=mps shallow=1.87
```

Without skip connections, the 24-block model is stuck at about 3.35 from step 250 on, while the 2-block model keeps improving. With them, the 24-block model beats the 2-block one: depth helps again once it can be trained.

On a CPU, use `--quick` (250 steps per run). Same machine, CPU only:

```text
RESULT before=3.35 after=2.08 seconds=106 device=cpu shallow=2.58
```

## The paper's numbers

Top-1 error on ImageNet validation, in percent with 10-crop testing (Table 2, page 5). Each ResNet has the same layers and parameters as the plain network next to it, plus the skip connections:

| | Plain | ResNet |
|---|---|---|
| 18 layers | 27.94 | 27.88 |
| 34 layers | 28.54 | 25.03 |

Going from 18 to 34 layers made the plain network worse and the residual network better. With skip connections the authors kept going: a 152-layer ResNet reached 21.43% top-1 error on the validation set (Table 3, page 6), and an ensemble of six residual networks of different depths, two of them 152 layers deep, won the ILSVRC 2015 classification task with 3.57% top-5 error on the test set (Table 5, pages 6 and 7).

## What broke next

The model can now be deep, but it only knows what it was trained on: about a megabyte of Shakespeare. To handle language in general, a model first has to learn from far more text, with no labels, and only then be taught a task. That is pre-training, in [step 06](../06-pretraining/) (GPT-1 and BERT).
