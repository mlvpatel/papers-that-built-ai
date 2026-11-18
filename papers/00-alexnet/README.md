# 00 AlexNet: a network that learns its own image filters

Paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), Krizhevsky, Sutskever, Hinton (University of Toronto, NeurIPS 2012). [PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

## What was broken

Before AlexNet, the best ImageNet systems averaged several classifiers: six sparse-coding models in the 2010 competition, and later two classifiers trained on Fisher vectors, the "SIFT + FVs" entry (Table 1 and text, page 7). They missed the right class in their top five guesses 28.2% and 25.7% of the time.

A neural network can learn its own features, but a plain one treats a photo as a flat list of numbers. Every unit in its first layer needs its own weight for every pixel, and nothing tells it that an edge in one corner is the same thing as an edge in another. Convolutional networks fix that, yet the authors write that they had been "prohibitively expensive" to apply at large scale to high-resolution images (page 2).

## The idea

A convolutional layer slides small filters over the image. Each filter is a handful of weights (3 by 3 pixels in the demo, 11 by 11 in AlexNet's first layer, page 4), and the same weights are used at every position. The paper names the two assumptions this builds in: nearby pixels depend on each other most, and a pattern means the same thing wherever it appears (page 1). Compared with fully connected layers of similar size, a CNN has far fewer connections and parameters, so it is easier to train, though the authors expect its best possible accuracy to be only slightly lower (page 1).

AlexNet stacked five convolutional layers and three fully connected ones, 60 million parameters in all (abstract, page 1). It trained for roughly 90 passes over 1.2 million photos, which took five to six days on two GTX 580 GPUs (page 7). It also used ReLU units, data augmentation and dropout (sections 3.1, 4.1 and 4.2).

The demo keeps only the convolution. Both models use ReLU, the same optimizer and the same batches; model B swaps the first fully connected layer for three stages like this one:

```python
def stage(c_in, c_out):  # AlexNet's pattern: convolution, ReLU, max pooling
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
```

## Run it

```bash
python papers/00-alexnet/demo.py
```

Run time on an Apple M5 Pro: 40 s for a full run on the GPU after the download, 76 s with `--quick` on the CPU.

The first run downloads CIFAR-10 (about 170 MB) into `data/`, which git ignores. Both models learn the 50,000 training photos for 10 epochs, with Adam, a batch of 128 (the paper's batch size, page 6), the same seed and the same batch order. Model A flattens each photo into 3,072 numbers and passes them through two fully connected layers of 512 units. Model B is the small CNN: three convolution stages, then one fully connected layer of 256 units. Accuracy is measured once, after the last epoch, on the 10,000 test photos. Guessing scores 10%.

Measured on an Apple M5 Pro (48 GB) GPU, 16 November 2026, after the download:

```text
50,000 training photos, 10,000 test photos
10 classes, so guessing scores 10%
A fully connected: 1,841,162 parameters
  A  epoch  1  train loss 1.639  1.2 s
  A  epoch  2  train loss 1.430  0.9 s
  A  epoch  3  train loss 1.316  1.0 s
  A  epoch  4  train loss 1.223  1.0 s
  A  epoch  5  train loss 1.144  0.9 s
  A  epoch  6  train loss 1.062  1.0 s
  A  epoch  7  train loss 0.991  1.0 s
  A  epoch  8  train loss 0.917  0.9 s
  A  epoch  9  train loss 0.838  0.9 s
  A  epoch 10  train loss 0.774  0.9 s
A fully connected: test accuracy 52.2%
B small CNN: 620,362 parameters
  B  epoch  1  train loss 1.388  2.5 s
  B  epoch  2  train loss 0.957  2.3 s
  B  epoch  3  train loss 0.788  2.5 s
  B  epoch  4  train loss 0.661  2.3 s
  B  epoch  5  train loss 0.560  2.5 s
  B  epoch  6  train loss 0.480  2.3 s
  B  epoch  7  train loss 0.389  2.2 s
  B  epoch  8  train loss 0.316  2.3 s
  B  epoch  9  train loss 0.248  2.5 s
  B  epoch 10  train loss 0.196  2.1 s
B small CNN: test accuracy 74.4%
100 CNN training batches: CPU 4.2 s, GPU 0.6 s
RESULT before=52.2 after=74.4 seconds=40 device=mps cpu_s=4.2 gpu_s=0.6
```

The CNN has a third of the parameters and gets 74.4% of the test photos right, against 52.2% for the fully connected model. The demo leaves out data augmentation and dropout, the two defences against overfitting AlexNet relied on (sections 4.1 and 4.2).

The line before RESULT is the other half of AlexNet's story: the same 100 training batches of the CNN take seven times as long on this laptop's CPU as on its GPU.

On a CPU, use `--quick` (4 epochs per model). Same machine, CPU only:

```text
RESULT before=52.0 after=73.6 seconds=76 device=cpu
```

## The paper's numbers

Error rates on the ILSVRC-2010 test set, 1,000 classes (Table 1, page 7). Top-5 error counts a photo as wrong when the right class is not among the model's five best guesses (page 2):

| Model | Top-1 error | Top-5 error |
|---|---|---|
| Sparse coding | 47.1% | 28.2% |
| SIFT + Fisher vectors | 45.7% | 25.7% |
| AlexNet | 37.5% | 17.0% |

In the ILSVRC-2012 competition, one network scored 18.2% top-5 error on the validation set. An average of seven, two of them first trained on the full ImageNet release of 15 million photos, reached 15.3% on the test set; the second-best entry scored 26.2% (Table 2 and text, page 7).

The paper used CIFAR-10 too, in two small side experiments. A four-layer CNN with ReLU reached 25% training error six times faster than the same network with tanh units (Figure 1, page 3), and a four-layer CNN reached 11% test error on CIFAR-10, 13% without the paper's response normalization (page 4).

## What broke next

AlexNet's weights, including the 96 first-layer filters shown in its Figure 3, were learned from labelled photos (pages 2 and 6). Learning a hidden layer at all needs a way to tell each hidden weight which way to move. That method is backpropagation, published in 1986: [step 01](../01-backprop/).
