# 45 landmark AI papers
*Compiled November 2026. Confidence: high for papers up to 2023, medium for 2024 to 2026.*

## Executive summary

Eight papers usually stand for modern AI: Word2Vec, the Transformer, GPT-1, BERT, the scaling laws, GPT-3, InstructGPT and Chain of Thought. Together they tell one story well: how text became a language model you can talk to. But that story has gaps. The biggest is between Word2Vec (2013) and the Transformer (2017): attention was invented in 2014, and the Transformer kept attention and threw out the RNN around it. The usual list also leaves out the papers that started deep learning (AlexNet, ResNet), the paper RLHF actually came from (Christiano 2017), the correction to Kaplan's scaling laws (Chinchilla 2022), and the follow-up to Chain of Thought (DeepSeek-R1, 2025: reasoning learned through reinforcement learning).

---

## Complete table: all 45 papers

Year = first public release (arXiv preprint or journal). Tiers: Famous (8) · Must-add (6) · Core = rest of the LLM story (11) · Beyond LLMs (12) · Practical (6) · Watch (2).

| # | Year | Paper | Authors (lab) | Area | Key idea | Tier |
|---|------|-------|---------------|------|----------|------|
| 1 | 1950 | [Computing Machinery and Intelligence](https://doi.org/10.1093/mind/LIX.236.433) | Turing (U Manchester) | Foundations | Can machines think? Proposes the Turing test | Beyond LLMs |
| 2 | 1986 | [Learning Representations by Back-propagating Errors](https://doi.org/10.1038/323533a0) | Rumelhart, Hinton, Williams (UCSD, CMU) | Foundations | Backpropagation: how neural networks learn | Beyond LLMs |
| 3 | 1997 | [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) | Hochreiter, Schmidhuber (TU Munich, IDSIA) | Sequences | Memory cells; main sequence model before Transformers | Beyond LLMs |
| 4 | 1998 | [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) | LeCun, Bottou, Bengio, Haffner (AT&T Labs) | Vision | Convolutional networks (LeNet) read handwriting | Beyond LLMs |
| 5 | 2003 | [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) | Bengio, Ducharme, Vincent, Jauvin (U Montréal) | Language | First neural language model with learned word vectors | Core |
| 6 | 2012 | [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Krizhevsky, Sutskever, Hinton (U Toronto) | Vision | Deep CNN on GPUs wins ImageNet; deep learning begins | Must-add |
| 7 | 2013 | [Word2Vec](https://arxiv.org/abs/1301.3781) | Mikolov, Chen, Corrado, Dean (Google) | Language | Words become coordinates | Famous |
| 8 | 2014 | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) | Goodfellow et al. (U Montréal) | Generation | Generator vs. discriminator; realistic generated images | Beyond LLMs |
| 9 | 2014 | [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215) | Sutskever, Vinyals, Le (Google) | Language | Encoder-decoder: read a sequence, write a sequence | Core |
| 10 | 2014 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | Bahdanau, Cho, Bengio (U Montréal) | Language | Attention is born: look back at every input word | Must-add |
| 11 | 2014 | [Adam](https://arxiv.org/abs/1412.6980) | Kingma, Ba (U Amsterdam, U Toronto) | Training | The default optimizer (as AdamW) for LLMs | Core |
| 12 | 2015 | [Human-Level Control through Deep RL (DQN)](https://doi.org/10.1038/nature14236) | Mnih et al. (DeepMind) | RL | One network learns Atari games from pixels | Beyond LLMs |
| 13 | 2015 | [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) | He, Zhang, Ren, Sun (Microsoft Research Asia) | Architecture | Skip connections make very deep nets trainable | Must-add |
| 14 | 2016 | [Mastering Go (AlphaGo)](https://doi.org/10.1038/nature16961) | Silver et al. (DeepMind) | RL | Neural nets + tree search beat a Go world champion | Beyond LLMs |
| 15 | 2016 | [Layer Normalization](https://arxiv.org/abs/1607.06450) | Ba, Kiros, Hinton (U Toronto) | Training | The normalization inside every Transformer block | Core |
| 16 | 2017 | [Sparsely-Gated Mixture of Experts](https://arxiv.org/abs/1701.06538) | Shazeer et al. (Google Brain) | Architecture | Send each token to a few expert sub-networks | Core |
| 17 | 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. (Google) | Architecture | The Transformer | Famous |
| 18 | 2017 | [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | Christiano, Leike, Brown, Martic, Legg, Amodei (OpenAI, DeepMind) | Alignment | Reward model from human A-vs-B choices: origin of RLHF | Must-add |
| 19 | 2017 | [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) | Schulman et al. (OpenAI) | RL | Stable RL algorithm used inside InstructGPT | Core |
| 20 | 2018 | [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Radford, Narasimhan, Salimans, Sutskever (OpenAI) | Language | Pre-train a Transformer on raw text, then fine-tune | Famous |
| 21 | 2018 | [BERT](https://arxiv.org/abs/1810.04805) | Devlin, Chang, Lee, Toutanova (Google) | Language | Bidirectional pre-training by predicting masked words | Famous |
| 22 | 2019 | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Radford et al. (OpenAI) | Language | Zero-shot tasks with no fine-tuning | Core |
| 23 | 2020 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan et al. (OpenAI) | Scaling | Loss falls predictably with size, data and compute | Famous |
| 24 | 2020 | [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Brown et al. (OpenAI) | Language | 175B parameters; learns tasks from examples in the prompt | Famous |
| 25 | 2020 | [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) | Lewis et al. (Facebook AI) | Engineering | Retrieve documents, then generate from them | Practical |
| 26 | 2020 | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | Ho, Jain, Abbeel (UC Berkeley) | Generation | Learn to remove noise step by step | Beyond LLMs |
| 27 | 2020 | [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) | Dosovitskiy et al. (Google) | Vision | Transformers on image patches replace CNNs | Beyond LLMs |
| 28 | 2021 | [CLIP](https://arxiv.org/abs/2103.00020) | Radford et al. (OpenAI) | Multimodal | Text and images in one shared space | Beyond LLMs |
| 29 | 2021 | [LoRA](https://arxiv.org/abs/2106.09685) | Hu et al. (Microsoft) | Engineering | Fine-tune by training small low-rank matrices | Practical |
| 30 | 2021 | [AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) | Jumper et al. (DeepMind) | Science | Protein structure from sequence; 2024 Nobel Prize | Beyond LLMs |
| 31 | 2021 | [Latent Diffusion (Stable Diffusion)](https://arxiv.org/abs/2112.10752) | Rombach et al. (LMU Munich, Runway) | Generation | Diffusion in a compressed space, cheap enough for a laptop | Beyond LLMs |
| 32 | 2022 | [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | Wei et al. (Google) | Reasoning | Write the steps before the answer | Famous |
| 33 | 2022 | [InstructGPT](https://arxiv.org/abs/2203.02155) | Ouyang et al. (OpenAI) | Alignment | RLHF makes GPT-3 follow instructions | Famous |
| 34 | 2022 | [Chinchilla](https://arxiv.org/abs/2203.15556) | Hoffmann et al. (DeepMind) | Scaling | About 20 tokens per parameter; corrects Kaplan | Must-add |
| 35 | 2022 | [FlashAttention](https://arxiv.org/abs/2205.14135) | Dao et al. (Stanford) | Engineering | Fast, memory-efficient exact attention | Practical |
| 36 | 2022 | [ReAct](https://arxiv.org/abs/2210.03629) | Yao et al. (Princeton, Google) | Agents | Alternate reasoning and tool actions | Practical |
| 37 | 2022 | [Constitutional AI](https://arxiv.org/abs/2212.08073) | Bai et al. (Anthropic) | Alignment | AI feedback guided by written principles | Practical |
| 38 | 2023 | [LLaMA](https://arxiv.org/abs/2302.13971) | Touvron et al. (Meta) | Language | Strong open-weights models | Core |
| 39 | 2023 | [Toolformer](https://arxiv.org/abs/2302.04761) | Schick et al. (Meta) | Agents | Model teaches itself when to call tools | Practical |
| 40 | 2023 | [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | Rafailov et al. (Stanford) | Alignment | Preference tuning without the RL step | Core |
| 41 | 2023 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | Lightman et al. (OpenAI) | Reasoning | Reward each reasoning step, not just the answer | Core |
| 42 | 2024 | [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) | Snell et al. (UC Berkeley, Google DeepMind) | Reasoning | Thinking longer can beat a bigger model | Core |
| 43 | 2025 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek-AI | Reasoning | Reasoning learned through RL on checkable answers | Must-add |
| 44 | 2025 | [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) | Muennighoff et al. (Stanford, UW, AI2) | Reasoning | Small fine-tune plus forced longer thinking | Watch |
| 45 | 2025 | [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) | Chen et al. (Anthropic) | Reasoning | Written chain of thought is not always faithful | Watch |

---

## Six to add first

This repo's 14 papers are the famous eight, five of these six, and backpropagation (row 2) as step 01. The sixth, Christiano et al., appears on [the step 09 page](../papers/09-instructgpt/) as the source of RLHF.

| # | Paper | Year | Why it belongs |
|---|-------|------|----------------|
| 1 | [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), Krizhevsky, Sutskever, Hinton | 2012 | Started the deep learning era: GPUs + big data + deep nets won ImageNet by a wide margin. #8 on Nature's list of the most-cited papers of the 21st century ([U of T](https://gro.utoronto.ca/?p=3548)). |
| 2 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Bahdanau, Cho, Bengio | 2014 | **Where attention was born** (as used in NLP). Fixed the bottleneck of squeezing a whole sentence into one vector. The Transformer took this idea and removed the RNN. The biggest gap in the usual list. |
| 3 | [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385), He, Zhang, Ren, Sun | 2015 | Skip connections made very deep networks trainable. Every Transformer block uses them. **#1 most-cited paper of the 21st century** in Nature's 2025 analysis ([Slashdot summary](https://tech.slashdot.org/story/25/04/18/1056230)). |
| 4 | [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741), Christiano, Leike, Brown, Martic, Legg, Amodei | 2017 | **Where RLHF came from**: train a reward model from human A-vs-B choices, then optimize against it. InstructGPT cites it as the source of its method ([InstructGPT](https://arxiv.org/abs/2203.02155)). |
| 5 | [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556), Hoffmann et al., DeepMind | 2022 | Corrected Kaplan 2020: scale data and parameters equally (about 20 tokens per parameter). A 70B model trained on 1.4T tokens beat 280B Gopher with the same compute; GPT-3 turned out to be badly undertrained ([Epoch AI replication](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)). |
| 6 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948), DeepSeek-AI (Nature 645, 633–638) | 2025 | The follow-up to Chain of Thought: models **learn** to reason through RL on checkable answers instead of being prompted to. Open weights and an open recipe. Peer reviewed in Nature ([The Register](https://www.theregister.com/2025/09/18/chinas_deepseek_ai_reasoning_research/)). Ranked #1 among 2025's most influential cs.CL papers ([Paper Digest](https://www.paperdigest.org/2025/09/most-influential-arxiv-computation-and-language-papers-2025-09-version/)). |

---

## Full timeline with the gaps filled

The eight famous papers are **bold**, with two rows holding two each; must-read additions are marked (must-add).

| Year | Paper | One-line idea |
|------|-------|---------------|
| 2003 | [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html), Bengio et al. | First neural language model with learned word vectors. Word2Vec builds on it. |
| 2012 | AlexNet (must-add) | Deep learning begins. |
| **2013** | **[Word2Vec](https://arxiv.org/abs/1301.3781), Mikolov et al.** | **Words become coordinates.** |
| 2014 | [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215), Sutskever, Vinyals, Le | Encoder-decoder: read a sequence, write a sequence. |
| 2014 | Bahdanau attention (must-add) | Attention is born. |
| 2014 | [Adam](https://arxiv.org/abs/1412.6980), Kingma & Ba | The optimizer (as AdamW) that trains nearly every LLM. |
| 2015 | ResNet (must-add) | Residual connections. |
| 2016 | [Layer Normalization](https://arxiv.org/abs/1607.06450), Ba, Kiros, Hinton | The normalization inside every Transformer block. |
| 2017 | [Outrageously Large Neural Networks (Mixture of Experts)](https://arxiv.org/abs/1701.06538), Shazeer et al. | Only part of the network runs for each token. Basis of Mixtral and DeepSeek-V3. |
| **2017** | **[Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al.** | **The Transformer.** |
| 2017 | Christiano et al. (must-add) | RLHF origin. |
| 2017 | [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347), Schulman et al. | The RL algorithm inside InstructGPT. |
| **2018** | **[GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) + [BERT](https://arxiv.org/abs/1810.04805)** | **Pre-training on raw text.** |
| 2019 | [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Radford et al. | Zero-shot tasks with no fine-tuning; the start of "just make it bigger". |
| **2020** | **[Scaling Laws](https://arxiv.org/abs/2001.08361) + [GPT-3](https://arxiv.org/abs/2005.14165)** | **Bigger is predictably better.** |
| **2022** | **[Chain of Thought](https://arxiv.org/abs/2201.11903), Wei et al.** | **Write the steps before the answer.** |
| **2022** | **[InstructGPT](https://arxiv.org/abs/2203.02155), Ouyang et al.** | **RLHF on GPT-3.** |
| 2022 | Chinchilla (must-add) | Fixes the scaling laws. |
| 2023 | [LLaMA](https://arxiv.org/abs/2302.13971), Touvron et al., Meta | Strong open-weights models; started the open model ecosystem. |
| 2023 | [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290), Rafailov et al. | RLHF without the RL step: a simple loss on preference pairs ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html)). |
| 2023 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), Lightman et al. | Reward each reasoning step, not just the final answer. |
| 2024 | [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314), Snell et al. | Thinking longer can beat a bigger model. The theory behind o1-style models. |
| 2025 | DeepSeek-R1 (must-add) | Reasoning learned through RL. |

Note: OpenAI's o1 (Sept 2024) started the reasoning model era, but OpenAI released a blog post and system card, not a research paper. DeepSeek-R1 is the paper that published the method.

---

## Landmarks outside the LLM story

The famous eight are all language. These made the other half of AI:

| Year | Paper | Area | Why |
|------|-------|------|-----|
| 1950 | [Computing Machinery and Intelligence](https://doi.org/10.1093/mind/LIX.236.433), Turing | Foundations | The Turing test; asks "can machines think?" |
| 1986 | [Learning Representations by Back-propagating Errors](https://doi.org/10.1038/323533a0), Rumelhart, Hinton, Williams | Foundations | Backpropagation: how every neural network learns. |
| 1997 | [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), Hochreiter & Schmidhuber | Sequences | The main sequence model before Transformers. |
| 1998 | [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791), LeCun et al. | Vision | Convolutional networks (LeNet). |
| 2014 | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Goodfellow et al. | Generation | Two networks compete; first realistic generated images. |
| 2015 | [Human-Level Control through Deep RL (DQN)](https://doi.org/10.1038/nature14236), Mnih et al. | RL | One network learns dozens of Atari games from pixels. |
| 2016 | [Mastering Go with Deep Neural Networks and Tree Search (AlphaGo)](https://doi.org/10.1038/nature16961), Silver et al. | RL | Beat a world champion at Go, about ten years earlier than most expected. |
| 2020 | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Ho, Jain, Abbeel | Images | The diffusion method behind Stable Diffusion, DALL·E 2, Imagen ([NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)). |
| 2020 | [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929), Dosovitskiy et al. | Vision | Transformers replace CNNs for images. |
| 2021 | [CLIP](https://arxiv.org/abs/2103.00020), Radford et al. | Multimodal | Text and images in one shared space; the link behind text-to-image. |
| 2021 | [Highly Accurate Protein Structure Prediction (AlphaFold 2)](https://doi.org/10.1038/s41586-021-03819-2), Jumper et al. | Science | Solved protein folding; part of the 2024 Nobel Prize in Chemistry. |
| 2021 | [High-Resolution Image Synthesis with Latent Diffusion](https://arxiv.org/abs/2112.10752), Rombach et al. | Images | Stable Diffusion: diffusion cheap enough to run on a laptop. |

---

## For building with LLMs (practical tier)

If you work as an AI engineer, these show up in daily work more than the theory papers:

| Year | Paper | What you use it for |
|------|-------|---------------------|
| 2020 | [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401), Lewis et al. | Ground answers in your own documents. |
| 2021 | [LoRA](https://arxiv.org/abs/2106.09685), Hu et al. | Fine-tune large models cheaply. |
| 2022 | [FlashAttention](https://arxiv.org/abs/2205.14135), Dao et al. | Fast, memory-efficient attention; the reason long context is affordable. |
| 2022 | [ReAct](https://arxiv.org/abs/2210.03629), Yao et al. | Reason + act loop: the basic pattern behind agents. |
| 2022 | [Constitutional AI](https://arxiv.org/abs/2212.08073), Bai et al., Anthropic | AI feedback in place of most human labels for safety training. |
| 2023 | [Toolformer](https://arxiv.org/abs/2302.04761), Schick et al. | Models that teach themselves when to call tools. |

---

## Too early to call (2025–2026)

No 2026 paper has clear landmark status yet. Citation counts lag by years, and the 2026 roundups disagree with each other ([Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/05/top-llm-research-papers-2026/), [AI Papers Academy](https://aipapersacademy.com/ai-research-trends-2026-h1/)). The main open questions are when extra "thinking" is worth the compute, and whether a model's written reasoning matches what it actually computes ([Apolo 2025 review](https://www.apolo.us/blog-posts/the-year-in-ai-best-of-2025-part-i-reasoning-models-llm-agents-and-more)). Papers to watch:
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) (2025), #2 on Paper Digest's 2025 cs.CL list.
- [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) (2025), Anthropic, on whether chain of thought is faithful.

---

## Method

Sources were web searches, publisher pages and an arXiv API check confirming that all 34 arXiv links resolve to the titles shown. Three questions shaped the list:
1. Which papers fill the gaps inside the usual LLM timeline?
2. Which non-LLM papers count as landmarks (by citation rank and field impact)?
3. Which 2024–2026 papers are becoming landmarks?

Ranking evidence: Nature's 2025 analysis of the most-cited 21st-century papers (ResNet #1, Transformer #7, AlexNet #8), Paper Digest influence rankings, and direct citations between papers (for example, InstructGPT citing Christiano 2017). Links point to arXiv or the publisher. On 18 November 2026 every link in this file was checked again: the 34 arXiv links return the papers named, and the seven DOI links for the pre-arXiv papers (Turing, backprop, LSTM, LeNet, DQN, AlphaGo, AlphaFold) resolve to the right titles and years. The 2025–2026 section relies on secondary roundups and has medium confidence.
