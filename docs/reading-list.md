# Reading list

The 14 papers this repo demonstrates, in the order the repo builds them: 12 steps, numbered 00 to 11, where steps 06 and 07 each cover two papers. Each title links to the paper's own page at the source, and the last column to its PDF; GPT-1 has no separate page, so its title links to the PDF too. The repo does not host copies of the papers.

| # | Step | Year | Paper | Authors | What it changed | PDF |
|---|---|---|---|---|---|---|
| 1 | 00 | 2012 | [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Krizhevsky, Sutskever, Hinton | Deep CNN on GPUs wins ImageNet; deep learning begins | [AlexNet PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| 2 | 01 | 1986 | [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0) | Rumelhart, Hinton, Williams | Backpropagation: how neural networks learn | [Backpropagation PDF](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) |
| 3 | 02 | 2013 | [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](https://arxiv.org/abs/1301.3781) | Mikolov, Chen, Corrado, Dean | Words become coordinates | [Word2Vec PDF](https://arxiv.org/pdf/1301.3781) |
| 4 | 03 | 2014 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | Bahdanau, Cho, Bengio | Attention is born: look back at every input word | [Bahdanau attention PDF](https://arxiv.org/pdf/1409.0473) |
| 5 | 04 | 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | The Transformer | [Transformer PDF](https://arxiv.org/pdf/1706.03762) |
| 6 | 05 | 2015 | [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) | He, Zhang, Ren, Sun | Skip connections make very deep nets trainable | [ResNet PDF](https://arxiv.org/pdf/1512.03385) |
| 7 | 06 | 2018 | [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Radford, Narasimhan, Salimans, Sutskever | Pre-train a Transformer on raw text, then fine-tune | [GPT-1 PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 8 | 06 | 2018 | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | Devlin, Chang, Lee, Toutanova | Bidirectional pre-training by predicting masked words | [BERT PDF](https://arxiv.org/pdf/1810.04805) |
| 9 | 07 | 2020 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan et al. | Loss falls predictably with size, data and compute | [Scaling laws PDF](https://arxiv.org/pdf/2001.08361) |
| 10 | 07 | 2022 | [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) | Hoffmann et al. | About 20 tokens per parameter; corrects Kaplan | [Chinchilla PDF](https://arxiv.org/pdf/2203.15556) |
| 11 | 08 | 2020 | [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | Brown et al. | 175B parameters; learns tasks from examples in the prompt | [GPT-3 PDF](https://arxiv.org/pdf/2005.14165) |
| 12 | 09 | 2022 | [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) | Ouyang et al. | RLHF makes GPT-3 follow instructions | [InstructGPT PDF](https://arxiv.org/pdf/2203.02155) |
| 13 | 10 | 2022 | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Wei et al. | Write the steps before the answer | [Chain of Thought PDF](https://arxiv.org/pdf/2201.11903) |
| 14 | 11 | 2025 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | DeepSeek-AI | Reasoning learned through RL on checkable answers | [DeepSeek-R1 PDF](https://arxiv.org/pdf/2501.12948) |

Steps follow build order, not date order: ResNet (2015) comes after the Transformer (2017) because the demo only needs skip connections once the model gets deep.

The full list of 45 papers, with the ones this repo leaves out, is in [all-45-papers.md](all-45-papers.md).
