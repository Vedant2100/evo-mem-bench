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

## What happened in the first, smaller test (36 test tasks per setup)

| Setup | Tasks completed (of 36) | Success rate | What the helper sees before starting (real example, clean-tomato chore) |
|---|---|---|---|
| No memory | 16 | 44% | Nothing. Result: took the tomato, skipped cleaning, put it on the table, then got stuck. |
| Retrieval of past attempts | about 19 | about 53% | Word-for-word records of earlier similar attempts (e.g. an apple-cleaning try), pasted in as they happened. |
| Letta-style note writer | about 19 | about 53% | A running set of notes: "If the fridge is closed, open it to check." "Find it, clean it in the sink, then place it." |
| Fixed hint sheet | about 22 | about 61% | One line, same for every task: "Take the target object, clean it using the appropriate sink or faucet, retrieve it, then place it in the requested receptacle." |
| Supervised note writer | about 23 | about 64% | A short to-do list from a past attempt: "Find the apple and take it. Clean the apple using the sinkbasin. Put the clean apple on the diningtable." |
| Reflection on past attempts | about 26 | about 72% | Lessons it wrote itself after practice: "Clean at the sink basin, not on a countertop." "Clean before putting it on the table." "Check the fridge and countertops first." |
| Trial-and-error picker (bandit) | 22 | 61% (vs 39% no-memory baseline in its own run) | Before each task it picks one of three ways of looking up past attempts (similar, most recent, or random) and learns from results which pick works. Its first 36 practice tasks are the only learning. |
| Trial-and-error picker using task details (contextual bandit) | 21 | 58% (same run) | Same picker, but it also looks at the kind of chore when choosing. |

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
- **What we did not test.** We did not train any model with reinforcement learning. All memory was given as prompts. The "trial-and-error" selection methods (bandits) ran only in the first, smaller test and got just 36 practice pulls.
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

## One example for each row of the table

All of these use the same chore: **"put a clean tomato on the side table"**. The "clean something" chore type has 6 test tasks, and the last column is how each setup did on all 6.

| Setup | What the helper was given before starting | Result on the tomato chore | Result on all 6 "clean something" tasks |
|---|---|---|---|
| No memory | Nothing. | Failed. It took the tomato from the fridge, skipped the cleaning, put it on the table, then got stuck. | 0 of 6 |
| Fixed hint sheet | One sentence: "Take the target object, clean it using the appropriate sink or faucet, retrieve it, then place it in the requested receptacle." | Succeeded. | 5 of 6 |
| Retrieval | Full records of earlier similar attempts (for example, an earlier apple-cleaning try), pasted in as they happened. | Succeeded. | 1 of 6 |
| Reflection | Short lessons it wrote about its own practice attempts, such as "clean at the sink basin, not on a countertop". | Succeeded in 7 turns. | 4 of 6 |
| Letta-style note writer | A running set of notes, for example "if the fridge is closed, open it to check", and "find it, clean it in the sink, then place it". | Succeeded. | 2 of 6 |
| Supervised note writer | A short to-do list from a past attempt: "Find the apple and take it. Clean the apple using the sinkbasin. Put the clean apple on the diningtable." | Succeeded. | 1 of 6 |

What the example shows: every memory setup got the tomato right, because even a one-line reminder to clean it first was enough. What it hides is that the setups differ a lot across the other five tasks. On this chore type the plain hint sheet did better than the more elaborate methods, and reflection's advantage showed up in the other chore types. This is one chore type with six tasks per setup, so it is an illustration, not a ranking.

## Is this the full list of methods we planned?

No. The original plan listed 10 setups for the screening stage, each with 10 practice and 10 test tasks per kind of chore. Because of the deadline we ran a smaller version: 6 practice and 6 test tasks per kind of chore.

- **Run:** no memory, fixed hint sheet, retrieval, reflection, Letta-style note writer, supervised note writer. We added the two trial-and-error pickers (bandits) later; they come from the project's wider list of methods.
- **Planned but not run at first (now run in the full comparison):** "most recent attempts", outcome-filtered retrieval, ExpeL-style rule extraction, and a probabilistic task-state method.
- **Only short working checks so far:** the evolutionary search and the methods adapted from other projects (A-MEM, ReasoningBank, Mem0 and others). They ran end to end but have not been run at full size.
- **Not an RL-trained memory model.** The bandits are the only RL-style method, and they only get 36 practice pulls.

So the results above show how the methods we ran compare. They do not say anything about the ones we left out.

---

## The full comparison (60 test tasks per setup)

This is the larger test from the original plan: 10 practice and 10 test tasks for each of the 6 kinds of chore, one run per setup. It ran in three batches. Every batch repeated "no memory" as its own baseline, and the helper's no-memory score moved a little between batches (40%, 43%, 47%), so each setup is compared with its own batch's baseline.

| Setup | Tasks completed (of 60) | Success rate | Gain over no memory in the same batch | What the helper sees before starting (real examples) |
|---|---|---|---|---|
| Supervised note writer | 39 | 65% | +18 points | A short to-do list written from past attempts, e.g. "Find the apple and take it. Clean it using the sinkbasin. Put it on the diningtable." |
| Reflection | 38 | 63% | +20 points | Lessons it wrote about its own earlier tries, e.g. "Clean at the sink basin, not on a countertop." |
| Bayesian task-state | 38 | 63% | +17 points | A running success score for this kind of chore ("about 92% of the last 10 tries worked") plus a step-by-step record of one that worked. |
| Fixed hint sheet | 36 | 60% | +20 points | One line, the same for every task: "Take the object, clean it at the sink, retrieve it, then place it." |
| ExpeL-style rules | 35 | 58% | +15 points | General rules pulled from many past tries, e.g. "Search likely storage places first; if unsure whether it's clean, clean it before placing it." |
| Outcome-filtered retrieval | 33 | 55% | +12 points | Word-for-word records of similar past tries, each marked SUCCESS or FAIL. |
| Recent attempts | 32 | 53% | +13 points | Word-for-word records of the most recent past tries, whatever they were about. |
| Retrieval | 30 | 50% | +10 points | Word-for-word records of the most similar past tries. |
| Letta-style note writer | 28 | 47% | 0 points | A running set of notes, rewritten as it goes. |
| No memory (three batches) | 24 / 28 / 26 | 40% / 47% / 43% | n/a | Nothing. |

The bandit methods from the earlier test (above) are not part of this table.

### What it shows
- Four setups finished close together at the top: the supervised note writer, reflection, Bayesian task-state and the fixed hint sheet (60% to 65%). One run per setup can't tell them apart.
- The fixed hint sheet, which costs nothing to make, did as well as the clever methods. That is a warning about how much the clever methods add.
- Methods that paste in whole records of past tries (retrieval, recent attempts, outcome-filtered) helped less than methods that boil the past down into short advice.
- The Letta-style writer gave no gain here.
- With 60 tasks per setup, gaps smaller than about 13 points could be chance.

### The remaining methods at full size

The ten methods adapted from other projects, plus evolutionary search, were then run at the same size (60 test tasks each). Again each batch repeated "no memory" as its own baseline (38% to 47% across batches).

| Setup | Tasks completed (of 60) | Success rate | Gain over no memory in the same batch |
|---|---|---|---|
| AWM | 40 | 67% | +25 points |
| AgentKB | 36 | 60% | +18 points |
| A-MEM | 34 | 57% | +18 points |
| Evolutionary retrieval search | 34 | 57% | +10 points |
| Mem0 | 33 | 55% | +17 points |
| ReasoningBank | 33 | 55% | +13 points |
| Lightweight Memory | 32 | 53% | +10 points |
| A-MEM with evolving notes | 31 | 52% | +13 points |
| ACE | 27 | 45% | +2 points |
| SkillWeaver | 27 | 45% | +2 points |

## All methods together (full size, 60 test tasks each unless noted)

| Rank | Setup | Success rate | Gain over its own no-memory baseline |
|---|---|---|---|
| 1 | AWM | 67% | +25 |
| 2 | Supervised note writer | 65% | +18 |
| 3 | Reflection | 63% | +20 |
| 3 | Bayesian task-state | 63% | +17 |
| 5 | Fixed hint sheet | 60% | +20 |
| 5 | AgentKB | 60% | +18 |
| 7 | ExpeL-style rules | 58% | +15 |
| 8 | A-MEM | 57% | +18 |
| 8 | Evolutionary retrieval search | 57% | +10 |
| 10 | Outcome-filtered retrieval | 55% | +12 |
| 10 | ReasoningBank | 55% | +13 |
| 10 | Mem0 | 55% | +17 |
| 13 | Recent attempts | 53% | +13 |
| 13 | Lightweight Memory | 53% | +10 |
| 15 | A-MEM with evolving notes | 52% | +13 |
| 16 | Retrieval | 50% | +10 |
| 17 | Letta-style note writer | 47% | 0 |
| 18 | ACE | 45% | +2 |
| 18 | SkillWeaver | 45% | +2 |
| (n/a) | No memory, across 8 batches | 38% to 47% | n/a |

The two trial-and-error pickers (bandits) were tested only in the smaller first test, on 36 tasks: 61% and 58% against 39% with no memory. They are not ranked here because of the different size.

### What to take from the whole set
- **Every method but three helped a bit or a lot.** The Letta-style writer, ACE and SkillWeaver added essentially nothing (0 to +2 points).
- **A group of about eight setups finished within a few points of each other** (AWM, supervised note writer, reflection, Bayesian task-state, the fixed hint sheet, AgentKB, ExpeL-style rules, A-MEM). With 60 tasks each and one run, we can't say which of these is really best.
- **The free fixed hint sheet matched the clever methods.** Most of the benefit seems to come from telling the helper the basic procedure (find it, clean or heat it, then put it away), not from remembering specific past tries.
- **Short advice beat whole transcripts.** Methods that boil the past into short rules or workflows tended to beat methods that paste in old records.
- **Heating chores stayed near 0% for every setup.** The helper is not good enough at that chore for memory to change the picture here.
- **No RL-trained memory model was tested.** The only RL-style methods were the two bandits, with 36 practice pulls.

### Limits
One run per setup, one helper model, one home simulator, and no-memory scores that moved by up to 9 points between batches. Differences under about 13 points could be chance. This is a first look to pick candidates for a proper repeat, not a final ranking.
