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

```bash
python papers/08-gpt3/demo.py
```

The first run downloads Pythia-1.4B (about 2.9 GB, pinned to one revision so results repeat), a base model trained only to predict the next token on 2020-era web text. It does two small tasks: label a review positive or negative, and rewrite a date as YYYY-MM-DD. Each item is asked twice: with only the task description, then with a few solved examples above it. The model, its weights and the items stay the same; only the prompt changes. An answer counts when the first line of the reply starts with it.

Measured on an Apple M5 Pro (48 GB) GPU, 16 November 2026:

```text
EleutherAI/pythia-1.4b on mps

Classify the review as positive or negative.  (4 examples in the few-shot prompt)
  want         zero-shot                few-shot
  positive     'the battery lasts two ' 'positive'
  negative     'it broke after a week ' 'negative'
  positive     'positive'               'positive'
  negative     'the hotel room smelled' 'negative'
  positive     'the review is positive' 'positive'
  negative     'i will never get back ' 'negative'
  positive     'i walked ten miles wit' 'positive'
  negative     'the app crashes every ' 'negative'
  zero-shot 1/8, few-shot 8/8

Convert the date to YYYY-MM-DD format.  (3 examples in the few-shot prompt)
  want         zero-shot                few-shot
  2021-03-05   'march 5, 2021'          '2021-03-05'
  1999-12-31   '1999-12-31'             '1999-12-31'
  1776-07-04   'july 4, 1776'           '1776-07-04'
  2024-02-29   'february 29, 2024'      '2024-02-29'
  1492-10-12   'october 12, 1492'       '1492-10-12'
  2030-06-01   'june 1, 2030'           '2030-06-01'
  2009-09-09   '2009-09-09'             '2009-09-09'
  1912-04-15   'april 15, 1912'         '1912-04-15'
  zero-shot 2/8, few-shot 8/8

zero-shot 3/16, few-shot 16/16
RESULT before=3 after=16 seconds=12 device=mps items=16
```

With only the description, the model mostly copies the input or keeps talking: 3 of 16. One more reply says "the review is positive", the right idea in the wrong format. With three or four examples above the question, it gets all 16, in the format the examples show.

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
