# 09 InstructGPT: train the model on what people prefer

Paper: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), Ouyang et al. (OpenAI, 2022). [PDF](https://arxiv.org/pdf/2203.02155)

*The demo on this page compares two different models, because no instruction-tuned copy of the base model's own weights can be downloaded without signing in. See Run it.*

## What was broken

GPT-3 was trained to predict the next token on web pages. The authors point out that this objective is different from the one users care about, "follow the user's instructions helpfully and safely", and call the language modeling objective misaligned (pages 1 and 2). Such models often make up facts, write biased or toxic text, or simply do not follow the instruction (page 1).

## The idea

Use people's judgments as the training signal, in three steps (Figure 2, page 3):

1. Labelers write good answers to real prompts, and GPT-3 is fine-tuned on them.
2. Labelers rank several model answers to the same prompt, best to worst. A reward model learns to predict their ranking.
3. The model is trained with reinforcement learning (PPO) to write answers the reward model scores highly.

The reward model works like a reviewer who has learned the team's taste and can now review every change. The team behind it was about 40 contractors (page 2). The method, reinforcement learning from human feedback, was first developed by Christiano et al. 2017 for simulated robots and Atari games (page 4; row 18 in [the list of 45 papers](../../docs/all-45-papers.md)).

## Run it

```bash
python papers/09-instructgpt/demo.py
```

Run time on an Apple M5 Pro: 13 s for a full run on the GPU, 30 s on the CPU (`--device cpu`).

The first run downloads two models (about 6.0 GB together, each pinned to one revision): Pythia-1.4B, trained only to predict the next token, and Qwen2.5-1.5B-Instruct, which was trained to follow instructions. They are two different models from two different labs, trained on different data, so the demo shows the kind of difference instruction training makes, not a controlled measurement of it. No instruction-tuned copy of Pythia's own weights can be downloaded without signing in. The paper's own comparison was controlled: its models share GPT-3's architecture and differ only in the human-feedback training (page 3).

Each instruction has a simple check, written before the first run: the haiku must be 3 lines, the tips exactly 3 numbered or bulleted items, the explanation one or two sentences. Both models get 120 tokens and greedy decoding.

Measured on an Apple M5 Pro (48 GB) GPU, 16 November 2026:

```text
### Write a haiku about debugging code.  (check: is_haiku)
--- EleutherAI/pythia-1.4b: fail ---
A:

I think you should use a different approach.
You can use a debugger to step through your code and see what is going on.
You can also use a debugger to step through your code and see what is going on.
You can use a debugger to step through your code and see what is going on.
You can use a debugger to step through your code and see what is going on.
You can use a debugger to step through your code and see what is going on.
You can use a debugger to step through
--- Qwen/Qwen2.5-1.5B-Instruct: pass ---
Debugging code,
Lines of code to unravel,
Finding the bug.

### List three tips for staying focused while studying.  (check: has_three_tips)
--- EleutherAI/pythia-1.4b: fail ---
1. Set a goal.

2. Set a deadline.

3. Set a time limit.

4. Set a reward.

5. Set a reward for the reward.

6. Set a reward for the reward.

7. Set a reward for the reward.

8. Set a reward for the reward.

9. Set a reward for the reward.

10. Set a reward for the reward.

11. Set a reward for the reward.

12. Set a reward for the reward.
--- Qwen/Qwen2.5-1.5B-Instruct: pass ---
1. Create a study schedule: Make sure to set aside specific times each day or week for studying and stick to it as closely as possible.
2. Eliminate distractions: Find a quiet place to study where you won't be interrupted by phone calls, social media, or other distractions.
3. Take breaks: It's important to take regular breaks during long periods of studying to avoid burnout and maintain focus.

### Explain what a neural network is to a 10-year-old in two sentences.  (check: at_most_two_sentences)
--- EleutherAI/pythia-1.4b: fail ---
A:

A neural network is a computer program that uses a set of rules to learn to perform a task.

A:

A neural network is a computer program that uses a set of rules to learn to perform a task.

A:

A neural network is a computer program that uses a set of rules to learn to perform a task.

A:

A neural network is a computer program that uses a set of rules to learn to perform a task.
--- Qwen/Qwen2.5-1.5B-Instruct: pass ---
A neural network is like a group of tiny brain cells working together to solve problems and make decisions. It's used to learn from data and improve its performance over time.

checks passed: Pythia 0/3, Qwen Instruct 3/3
RESULT before=0 after=3 seconds=13 device=mps checks=3
```

Pythia treats each instruction as the start of a forum page: it answers as "A:", then repeats one sentence until it runs out of tokens, and asked for three tips it lists twelve. Qwen Instruct does what was asked all three times. The haiku check counts lines, not syllables, and Qwen's three lines have 4, 7 and 4 syllables instead of 5, 7 and 5.

On the CPU (`--device cpu`, 27 November 2026), Qwen Instruct again passed all 3 checks and Pythia passed 1. Pythia's haiku reply was "A:", one sentence, and a second sentence repeated along a single line. That is three lines, so the check passes it, though it is no haiku: the check counts lines and nothing else. Greedy decoding always takes the most likely next token, and the CPU rounds its arithmetic slightly differently from the GPU, so where two tokens are nearly tied the reply can change.

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
