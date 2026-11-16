# 10 Chain of Thought: write the steps before the answer

Paper: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), Wei et al. (Google, 2022). [PDF](https://arxiv.org/pdf/2201.11903)

## What was broken

Bigger models and few-shot prompts helped on most tasks but not on multi-step problems. The authors write that "scaling up model size alone has not proved sufficient" for arithmetic, commonsense and symbolic reasoning, and that the few-shot prompting of step 08 "works poorly on tasks that require reasoning abilities" (page 2). A math word problem asks the model to produce a number in one go, with every intermediate step done silently.

## The idea

Let the model show its working. Each example in the prompt becomes a triple: the question, a chain of thought written in plain language, and the answer (page 2). The paper's first example (Figure 1, page 1):

```text
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis
   balls. 5 + 6 = 11. The answer is 11.
```

Writing the steps breaks a problem into parts and leaves a trace you can read when an answer is wrong (page 3), much like adding print statements to a function. Extra tokens alone are not the reason: when the model writes a row of dots as long as the equation it needs, instead of the steps, it does about as well as with no help at all (page 6; Table 6, page 23). The authors wrote eight such examples by hand and used the same eight for every math benchmark except one (pages 3 and 4).

## Run it

```bash
python papers/10-chain-of-thought/demo.py
```

The first run downloads Qwen2.5-1.5B-Instruct (about 3.1 GB, pinned to one revision so results repeat). The same model answers the same 12 word problems twice: first told to give only the final number, then told to reason step by step and put the answer in `\boxed{}`. The answer keys are computed in code, and decoding is greedy, so a rerun on the same machine prints the same text. The problems were written on 15 November 2026 for this test and not changed after.

Measured on an Apple M5 Pro (48 GB) GPU, 16 November 2026:

```text
Qwen/Qwen2.5-1.5B-Instruct on mps; 12 word problems
  direct   1  want 24  got 18.0  wrong  (3 tokens)
  direct   2  want 15  got 75.0  wrong  (3 tokens)
  direct   3  want 162  got 108.0  wrong  (4 tokens)
  direct   4  want 36  got 48.0  wrong  (3 tokens)
  direct   5  want 54  got 60.0  wrong  (3 tokens)
  direct   6  want 75  got 75.0  right  (3 tokens)
  direct   7  want 6  got 8.0  wrong  (2 tokens)
  direct   8  want 9368  got 680.0  wrong  (4 tokens)
  direct   9  want 24  got 24.0  right  (3 tokens)
  direct  10  want 9  got 912.0  wrong  (4 tokens)
  direct  11  want 72  got 240.0  wrong  (4 tokens)
  direct  12  want 46  got 130.0  wrong  (4 tokens)
direct: 2/12 right, 40 tokens, 1 s
  steps    1  want 24  got 24.0  right  (319 tokens)
  steps    2  want 15  got 15.0  right  (279 tokens)
  steps    3  want 162  got 162.0  right  (392 tokens)
  steps    4  want 36  got 36.0  right  (308 tokens)
  steps    5  want 54  got 54.0  right  (329 tokens)
  steps    6  want 75  got 75.0  right  (319 tokens)
  steps    7  want 6  got 6.0  right  (290 tokens)
  steps    8  want 9368  got 9474.0  wrong  (655 tokens)
  steps    9  want 24  got 24.0  right  (513 tokens)
  steps   10  want 9  got 9.0  right  (451 tokens)
  steps   11  want 72  got 72.0  right  (365 tokens)
  steps   12  want 46  got 46.0  right  (400 tokens)
steps : 11/12 right, 4,620 tokens, 63 s
...
RESULT before=2 after=11 seconds=69 device=mps direct_tokens=40 step_tokens=4620
```

Told to answer at once, the model gets 2 of 12. Told to write the steps, it gets 11 of 12, using about 385 tokens per problem instead of 3. The one miss is problem 8, the sum of numbers up to 200 divisible by 3 or 5: it answered 9,474 instead of 9,368. Before the RESULT line, the demo prints its full step-by-step reply to the last problem, a quadratic solved line by line.

The paper puts worked examples in the prompt; the demo asks in words, which an instruction-trained model (step 09) understands.

## The paper's numbers

GSM8K, a benchmark of grade-school math word problems (Figure 2, page 2):

| Model | Solve rate |
|---|---|
| PaLM 540B, ordinary prompting | 18% |
| GPT-3 175B, fine-tuned for the task | 33% |
| Previous best | 55% |
| PaLM 540B, chain-of-thought prompting | 57% |

It only works in big models. Chain-of-thought prompting gives no gain for small models and helps only at around 100 billion parameters (page 4), and it actually hurts most models under 10 billion (page 16). Smaller models wrote fluent but illogical chains (page 4).

It is not only math. With PaLM 540B, StrategyQA went from 68.6% to 77.8% and sports understanding from 80.5% to 95.4% (Table 4, page 22).

## What broke next

The authors list the limits themselves: "there is no guarantee of correct reasoning paths", and writing chains of thought by hand, cheap for a few prompt examples, could cost too much for fine-tuning (page 9). DeepSeek-R1 puts it more sharply: reasoning built this way is "heavily contingent upon extensive human-annotated demonstrations", and a model copying human reasoning is capped by the examples it was given (DeepSeek-R1, pages 1 and 2). [Step 11](../11-deepseek-r1/) lets the model learn to reason by checking its answers.
