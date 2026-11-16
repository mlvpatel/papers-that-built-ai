# 08 GPT-3: show the model a few examples instead of retraining it

Paper: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), Brown et al. (OpenAI, 2020). [PDF](https://arxiv.org/pdf/2005.14165)

## What was broken

After step 06, every new task still meant fine-tuning. The authors write that strong performance on a task "typically requires fine-tuning on a dataset of thousands to hundreds of thousands of examples specific to that task" (page 3). Someone has to collect and label that data again for each new task, and many useful tasks never get a dataset at all.

## The idea

Put the examples in the prompt. GPT-3 reads a task description and a few solved examples, then completes the next one, with no weight updates at all (pages 3, 5 and 6, Figure 2.1 on page 7). The paper calls this in-context learning. It is the difference between retraining a colleague for every job and handing them a sheet with two filled-in rows.

| Setting | What goes in the prompt |
|---|---|
| Zero-shot | A description of the task only |
| One-shot | The description and one solved example |
| Few-shot | The description and as many examples as fit, typically 10 to 100 |

The model is GPT-2's design with some sparse attention layers added (page 8), made very large. GPT-3 has 175 billion parameters, 10 times more than any earlier non-sparse language model (abstract, page 1), and the paper trained eight sizes from 125 million to 175 billion, all on 300 billion tokens (Table 2.1, page 8).

## Run it

Demo coming in v1.1. It will give Pythia-1.4B, a small open model trained only to predict the next token, two tasks (labelling the sentiment of a sentence, and rewriting a date in a fixed format), first with no examples in the prompt and then with a few. The model, its weights and the test items stay the same; only the prompt changes.

## The paper's numbers

GPT-3 175B on basic arithmetic, where the answer must be exactly right (Table 3.9, page 23):

| Task | Zero-shot | One-shot | Few-shot |
|---|---|---|---|
| 2-digit addition | 76.9% | 99.6% | 100.0% |
| 3-digit addition | 34.2% | 65.5% | 80.4% |
| 2-digit multiplication | 19.8% | 27.4% | 29.2% |

Size matters as much as the examples. The 13-billion-parameter model solves 2-digit addition and subtraction only about half the time, even few-shot (page 22). Across 42 benchmarks, few-shot accuracy climbs faster with model size than zero-shot accuracy does (Figure 1.3, page 5).

On TriviaQA, answering from memory with no documents, GPT-3 scores 64.3% zero-shot, 68.0% one-shot and 71.2% few-shot, and the paper calls the last state of the art against fine-tuned models in the same closed-book setting (page 5).

## What broke next

A bigger model is not a more helpful one. The InstructGPT paper opens with exactly that: "Making language models bigger does not inherently make them better at following a user's intent," and such models can be "untruthful, toxic, or simply not helpful" (InstructGPT, abstract, page 1). [Step 09](../09-instructgpt/) trains the model on what people actually want.
