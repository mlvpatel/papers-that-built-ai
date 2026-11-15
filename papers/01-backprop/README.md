# 01 Backpropagation: how a network learns its hidden layers

Paper: [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0), Rumelhart, Hinton, Williams. Nature 323, 533–536 (October 1986). [PDF](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf), hosted on Geoffrey Hinton's site.

## What was broken

A network with only an input layer and an output layer draws a single straight line, or a flat plane, through its inputs. Many problems need a bent boundary, and that takes a hidden layer in between. The trouble was training it: nobody had a reliable way to tell a hidden unit how to change, because its correct value never appears in the data.

XOR is the smallest such problem. It outputs 1 when exactly one of two inputs is 1. Plot the four inputs on a square: the two 1s sit on one diagonal and the two 0s on the other, so no single straight line puts the 1s on one side and the 0s on the other. The best a straight line can do is three points out of four.

## The idea

Run the network forward, measure the error at the output, and send that error backwards through the network, layer by layer, to work out how much each weight contributed to it. Then nudge every weight to shrink the error. In the paper's words, the procedure "repeatedly adjusts the weights of the connections in the network so as to minimize a measure of the difference between the actual output vector of the net and the desired output vector", and as a result the hidden units "come to represent important features of the task domain" (abstract).

In PyTorch the whole backward pass is one line:

```python
loss.backward()  # backpropagation: the error flows back through every layer
```

## Run it

```bash
python papers/01-backprop/demo.py
```

Model A has no hidden layer. Model B has a hidden layer of four units. Both start from the same seed and train for 2,000 steps on the four XOR points. The table shows each model's output for each input; above 0.5 counts as 1.

Measured on an Apple M5 Pro (48 GB), CPU, 15 November 2026:

```text
input  target | A: no hidden layer | B: hidden layer
0 0      0    |        0.50        |      0.00
0 1      1    |        0.50        |      1.00
1 0      1    |        0.50        |      1.00
1 1      0    |        0.50        |      0.00
RESULT before=50 after=100 seconds=1 device=cpu
```

Model A gives up: with no line that works, its best answer under this loss is 0.50 for every input, which scores 2 of 4. Model B gets all four right. The run takes about a second on any machine, so `--quick` is the same as the full run.

## The paper's numbers

The 1986 paper shows backpropagation on its own example tasks in its figures. The copy linked above is a scan with no text layer, so this page could not machine-check those figures and quotes only the abstract from the publisher's page. Its key claim is the one this demo shows: "The ability to create useful new features distinguishes back-propagation from earlier, simpler methods such as the perceptron-convergence procedure."

## What broke next

Backpropagation can train any network, but a network that sees words as plain symbols has no idea that *king* and *queen* are related. [Step 02](../02-word2vec/) trains a vector for every word so that related words sit close together.
