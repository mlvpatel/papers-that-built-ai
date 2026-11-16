# 09 InstructGPT: train the model on what people prefer

Paper: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), Ouyang et al. (OpenAI, 2022). [PDF](https://arxiv.org/pdf/2203.02155)

## What was broken

GPT-3 was trained to predict the next token on web pages. The authors point out that this objective is different from the one users care about, "follow the user's instructions helpfully and safely", and call the language modeling objective misaligned (pages 1 and 2). Such models often make up facts, write biased or toxic text, or simply do not follow the instruction (page 1).

## The idea

Use people's judgments as the training signal, in three steps (Figure 2, page 3):

1. Labelers write good answers to real prompts, and GPT-3 is fine-tuned on them.
2. Labelers rank several model answers to the same prompt, best to worst. A reward model learns to predict their ranking.
3. The model is trained with reinforcement learning (PPO) to write answers the reward model scores highly.

The reward model works like a reviewer who has learned the team's taste and can now review every change. The team behind it was about 40 contractors (page 2). The method, reinforcement learning from human feedback, was first developed by Christiano et al. 2017 for simulated robots and Atari games (page 4; row 18 in [the list of 45 papers](../../docs/all-45-papers.md)).

## Run it

Demo coming in v1.1. It will give the same three instructions to two different models: Pythia-1.4B, trained only to predict the next token, and Qwen2.5-1.5B-Instruct, which was trained to follow instructions. They are two different models from two different labs, trained on different data, so the demo shows the kind of difference instruction training makes, not a controlled measurement of it. The paper's own comparison was controlled: its models share GPT-3's architecture and differ only in the human-feedback training (page 3).

## The paper's numbers

Labelers compared answers on held-out prompts from API customers (pages 3 and 11):

| Comparison | How often labelers preferred InstructGPT |
|---|---|
| InstructGPT 175B vs. GPT-3 175B | 85 ± 3% |
| InstructGPT 175B vs. GPT-3 175B with a few-shot prompt | 71 ± 4% |

Outputs from the 1.3-billion-parameter InstructGPT were preferred to those of the 175-billion-parameter GPT-3, a model with over 100 times more parameters (abstract, page 1; Figure 1, page 2).

| Measure | GPT-3 | InstructGPT | Source |
|---|---|---|---|
| Made-up facts on tasks where the answer is in the input | 41% | 21% | page 3 |
| Truthful and informative answers on TruthfulQA | | about twice as often | page 3 |
| Toxic outputs when asked to be respectful | | about 25% fewer | page 3 |

It did not reduce bias on the Winogender and CrowS-Pairs tests (page 3). The training was cheap next to pre-training: 60 petaflop/s-days for the 175B model's PPO run against 3,640 for GPT-3 itself (page 17).

## What broke next

A model that follows instructions still struggles with problems that take several steps. The Chain of Thought paper notes that "scaling up model size alone has not proved sufficient" for arithmetic, commonsense and symbolic reasoning, and that ordinary few-shot prompting "works poorly on tasks that require reasoning abilities" (Chain of Thought, page 2). [Step 10](../10-chain-of-thought/) has the model write out its steps.
