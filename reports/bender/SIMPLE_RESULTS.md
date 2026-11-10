# Does memory help an AI helper do household chores? A plain-language summary

## What we asked

An AI helper is given household tasks in a simulated home, for example "put a cleaned apple on the table" or "find the lamp and look at a book under it". If the helper can remember what happened on earlier attempts, does it get better at the next ones?

## What we did

- We used one AI model (Qwen3-32B) as the helper. It was **not** retrained. Any memory was given to it as written notes.
- We compared six setups. Each helper was tested on the same 36 tasks: 6 tasks in each of 6 kinds of chore.
  1. **No memory:** the helper starts fresh every time.
  2. **Fixed hint sheet:** a short list of general tips, the same for every task.
  3. **Retrieval:** the helper is shown notes from the most similar past attempts.
  4. **Reflection:** after each attempt the helper writes down what went wrong and how to do better, and reads those notes next time.
  5. **Letta-style note writer:** the helper keeps and rewrites a running set of notes, in the style described in Letta's blog posts.
  6. **Supervised note writer:** a separate writer is prompted to summarise lessons from past attempts.
- Before each memory setup was tested, it got 6 practice tasks per kind of chore.

## What happened

| Setup | Tasks completed (of 36) | Success rate |
|---|---|---|
| No memory | 16 | 44% |
| Retrieval of past attempts | about 19 | about 53% |
| Letta-style note writer | about 19 | about 53% |
| Fixed hint sheet | about 22 | about 61% |
| Supervised note writer | about 23 | about 64% |
| Reflection on past attempts | about 26 | about 72% |

The counts marked "about" were worked out from the reported differences, not read from the raw counts.

## What we take from it

- Every memory setup did better than no memory in this test.
- Reflection did best. It completed roughly 10 more tasks out of 36 than no memory.
- The Letta-style note writer did worse than the plain fixed hint sheet.
- Even the best setup fails more than a quarter of the time.

## How much to trust it

- **Small test.** There are only 36 tasks per setup and we ran each setup once. Differences smaller than about 20 points could be down to chance. Only reflection's lead is large enough to stand out.
- **One kind of chore never worked.** Heating things (for example "heat a potato and put it on the counter") failed almost every time, with or without memory, so it tells us nothing.
- **Not strict.** We lowered our own pass bar for the helper's baseline skill to meet a deadline, so treat this as a first look, not a final result.
- **What we did not test.** We did not train any model with reinforcement learning. All memory was given as prompts. A trial of "trial-and-error" selection methods (bandits) was still finishing when this was written.
- **One helper, one home simulator.** Results may not carry over to other models or settings.

## Where the details are

Full numbers and code are in the same folder as this file (`reports/bender/`) in the repository `Vedant2100/evo-mem-bench`, branch `bender-screen-design`.

## A real example

The chore: **"put a clean tomato on the side table"**. This is one of six held-back test tasks of the "clean something and put it away" kind.

- **Without memory**, the helper found the tomato in the fridge but put it on the side table without cleaning it first. It then got stuck looking around until its 40 turns ran out. Across the six tasks of this kind it completed **0 of 6**.
- **With reflection**, the helper had first practised on six similar tasks (for example, cleaning an apple) and written itself short lessons from those attempts. In paraphrase, they said things like:
  - "Cleaning has to be done at the sink basin, not on a countertop."
  - "Clean the item *before* putting it down on the table."
  - "Check likely places like the fridge and countertops first."
- On the tomato task it then went straight through: open the fridge, take the tomato, go to the sink, clean it, go to the side table, put it down. That took **7 turns**.
- Across the six tasks of this kind, reflection completed **4 of 6**.

This is one chore type with six tasks, so it shows what reflection can do, not how often it will.
