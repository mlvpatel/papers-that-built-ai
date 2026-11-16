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

Demo coming in v1.1. It will ask Qwen2.5-1.5B-Instruct the same 12 word problems twice: once told to give only the answer, once told to work step by step. The paper puts worked examples in the prompt; the demo asks in words, which an instruction-trained model (step 09) understands.

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
