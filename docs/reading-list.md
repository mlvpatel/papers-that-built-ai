# Reading list

The 14 papers this repo demonstrates, in the order the repo builds them: 12 steps, numbered 00 to 11, where steps 06 and 07 each cover two papers. Each row links the paper's own page and its PDF at the source. The repo does not host copies of the papers.

| # | Step | Year | Paper | Authors | What it changed | Read |
|---|---|---|---|---|---|---|
| 1 | 00 | 2012 | ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | Krizhevsky, Sutskever, Hinton | Deep CNN on GPUs wins ImageNet; deep learning begins | [page](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| 2 | 01 | 1986 | Learning representations by back-propagating errors | Rumelhart, Hinton, Williams | Backpropagation: how neural networks learn | [page](https://doi.org/10.1038/323533a0), [PDF](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) |
| 3 | 02 | 2013 | Efficient Estimation of Word Representations in Vector Space (Word2Vec) | Mikolov, Chen, Corrado, Dean | Words become coordinates | [page](https://arxiv.org/abs/1301.3781), [PDF](https://arxiv.org/pdf/1301.3781) |
| 4 | 03 | 2014 | Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau, Cho, Bengio | Attention is born: look back at every input word | [page](https://arxiv.org/abs/1409.0473), [PDF](https://arxiv.org/pdf/1409.0473) |
| 5 | 04 | 2017 | Attention Is All You Need | Vaswani et al. | The Transformer | [page](https://arxiv.org/abs/1706.03762), [PDF](https://arxiv.org/pdf/1706.03762) |
| 6 | 05 | 2015 | Deep Residual Learning for Image Recognition (ResNet) | He, Zhang, Ren, Sun | Skip connections make very deep nets trainable | [page](https://arxiv.org/abs/1512.03385), [PDF](https://arxiv.org/pdf/1512.03385) |
| 7 | 06 | 2018 | Improving Language Understanding by Generative Pre-Training (GPT-1) | Radford, Narasimhan, Salimans, Sutskever | Pre-train a Transformer on raw text, then fine-tune | [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 8 | 06 | 2018 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin, Chang, Lee, Toutanova | Bidirectional pre-training by predicting masked words | [page](https://arxiv.org/abs/1810.04805), [PDF](https://arxiv.org/pdf/1810.04805) |
| 9 | 07 | 2020 | Scaling Laws for Neural Language Models | Kaplan et al. | Loss falls predictably with size, data and compute | [page](https://arxiv.org/abs/2001.08361), [PDF](https://arxiv.org/pdf/2001.08361) |
| 10 | 07 | 2022 | Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann et al. | About 20 tokens per parameter; corrects Kaplan | [page](https://arxiv.org/abs/2203.15556), [PDF](https://arxiv.org/pdf/2203.15556) |
| 11 | 08 | 2020 | Language Models are Few-Shot Learners (GPT-3) | Brown et al. | 175B parameters; learns tasks from examples in the prompt | [page](https://arxiv.org/abs/2005.14165), [PDF](https://arxiv.org/pdf/2005.14165) |
| 12 | 09 | 2022 | Training language models to follow instructions with human feedback (InstructGPT) | Ouyang et al. | RLHF makes GPT-3 follow instructions | [page](https://arxiv.org/abs/2203.02155), [PDF](https://arxiv.org/pdf/2203.02155) |
| 13 | 10 | 2022 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei et al. | Write the steps before the answer | [page](https://arxiv.org/abs/2201.11903), [PDF](https://arxiv.org/pdf/2201.11903) |
| 14 | 11 | 2025 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI | Reasoning learned through RL on checkable answers | [page](https://arxiv.org/abs/2501.12948), [PDF](https://arxiv.org/pdf/2501.12948) |

Steps follow build order, not date order: ResNet (2015) comes after the Transformer (2017) because the demo only needs skip connections once the model gets deep.

The full list of 45 papers, with the ones this repo leaves out, is in [all-45-papers.md](all-45-papers.md).
