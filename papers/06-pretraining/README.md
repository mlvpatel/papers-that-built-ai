# 06 Pre-training: learn the language first, then the task

Papers: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1), Radford, Narasimhan, Salimans, Sutskever (OpenAI, 2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Devlin, Chang, Lee, Toutanova (Google, 2018). [PDF](https://arxiv.org/pdf/1810.04805)

## What was broken

Step 05's model can be deep, but it learns only from the examples it is given. Labeled examples cost people time to make, so some tasks have very few: in GLUE, the benchmark both papers use, RTE has 2.5k labeled training examples while MNLI has 392k (BERT, Table 1, page 6). A model trained on a small set learns those examples rather than the language, and every new task starts from nothing.

## The idea

First train the model on a large amount of raw text with no labels at all, by predicting the next word. That teaches it how the language works. Then train it on the task. GPT-1 pre-trained on BooksCorpus, over 7,000 unpublished books (page 4). BERT kept the recipe and changed the pre-training task: it hides some words and predicts them from both sides, where GPT-1 reads only left to right (pages 1 and 2).

In the demo the pre-training objective is one line, the same next-character loss as steps 04 and 05:

```python
# predict the next character
loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
```

## Run it

```bash
python papers/06-pretraining/demo.py
```

The test: given a line of Shakespeare and the same words in shuffled order, which one is real? Lines are lowercased with punctuation removed, so word order is the only clue. Each model picks the order it finds more likely, meaning the lower next-character loss. Model A learns from the 300 task lines only. Model B, the same network from the same seed with the same 3,000 steps, learns from the whole text with every test line cut out. Chance is 50%.

Measured on an Apple M5 Pro (48 GB) GPU, 15 November 2026:

```text
500 test pairs; chance is 50%
  real:     by putting on the destined livery
  shuffled: on the putting destined livery by
A reads 12,607 characters; B reads 1,048,128
  A  step    0  loss 3.516
  A  step 1000  loss 0.184
  A  step 2000  loss 0.159
  A  step 2999  loss 0.138
A, the 300 task lines only:  picks the real line 59.8%
  B  step    0  loss 3.518
  B  step 1000  loss 1.507
  B  step 2000  loss 1.373
  B  step 2999  loss 1.325
B, pre-trained on the text:  picks the real line 91.4%
RESULT before=59.8 after=91.4 seconds=155 device=mps
```

Model A learned its 300 lines almost by heart (training loss 0.14) and still picks the real line only 59.8% of the time. Model B never saw a test line and picks it 91.4% of the time.

On a CPU, use `--quick` (600 steps per model). The gap is smaller with less pre-training:

```text
RESULT before=63.8 after=72.6 seconds=70 device=cpu
```

### What was tried first

Two earlier versions followed GPT-1's second stage more closely: fine-tune a classifier on 300 labeled lines. Settings were fixed before the first run, and the bar was a 10-point gap.

| Task | Without pre-training | With pre-training |
|---|---|---|
| Which of the five busiest speakers says a line | 26.2% | 30.6% |
| Real vs. shuffled line, fine-tuned classifier | 54.8% | 52.6% |

Neither cleared the bar. A check on the training lines showed the pre-trained model already preferred the real word order 93% of the time with no labels, so pre-training had worked and the small fine-tuning recipe threw the knowledge away. BERT's authors saw fine-tuning of their large model become unstable on small datasets too (page 6). The test lines above were drawn fresh, leaving out the earlier attempt's test lines.

## The paper's numbers

GPT-1 ran the same comparison at full scale. The same Transformer, scored as an unweighted average over eight benchmark tasks, reached 74.7 with pre-training and 59.9 without it (Table 5, page 8). On the grammar task CoLA the gap was 45.4 against 18.9. GPT-1 improved the state of the art on 9 of the 12 tasks it tried (page 1), and it also measured abilities with no fine-tuning at all, for example scoring sentences by average log-probability as this demo does (Figure 2, page 7; method on page 8).

BERT's GLUE test averages, leaving out the WNLI task (Table 1, page 6):

| Model | GLUE average |
|---|---|
| OpenAI GPT | 75.1 |
| BERT-Base, the same size as GPT | 79.6 |
| BERT-Large | 82.1 |

## What broke next

Pre-training works, so how big should the model be, and how much text should it read? Kaplan et al. found that loss falls in a predictable curve as models, data and compute grow, and Chinchilla later corrected how to split a fixed budget between model size and data. That is [step 07](../07-scaling/).
