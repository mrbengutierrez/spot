1. Problem they’re poking at

Everyone worries about LLM hallucinations, so lots of metrics have been invented to measure how bad they are (ROUGE-ish overlap, BERTScore-ish similarity, task-specific things like Q², NLI classifiers like Critic, LLM-as-judge, etc.).

But almost nobody has stopped to ask:

Do these metrics actually agree with humans?

Do they work outside the dataset or model they were built/tested on?

Do they get better when the underlying LLM gets bigger or better?

Do they behave the same for different decoding strategies (greedy vs sampling)?

This paper says: let’s test that, properly, and at scale.

2. What they actually evaluated

They create a big evaluation grid:

4 datasets:

Begin (3 knowledge-grounded dialogue sets with human hallucination labels)

HaluEval (hallucination benchmark with dialog + QA, also with labels)

TruthfulQA (factual QA, but they generate the answers themselves)

FaithDial (knowledge-grounded dialog, they also generate answers)

Only Begin and HaluEval come with human labels for hallucination, so those two are where they can really check “does the metric match humans?”

37 models from 5 families (OPT, Llama, OLMo, Phi, Gemma), small to big, often both base and instruction-tuned.

5 decoding methods: greedy, beam, ancestral, top-k, top-p.

6 kinds of hallucination metrics:

n-gram / overlap: ROUGE-L, SacreBLEU, Knowledge-F1

semantic similarity: BERTScore, Knowledge-BERTScore

UniEval (pretrained evaluators for consistency/groundedness)

Q² (generate questions from output, answer them, check against source)

Critic (NLI-style classifier that says “this is unfaithful”)

LLM-as-judge (GPT-4) + an ensemble that combines several signals

So: many metrics × many models × many decoding styles × multiple datasets.

That’s the whole paper: stress-test the metrics themselves.

3. Their four main findings
Finding 1: Most metrics don’t reliably match humans. GPT-4 does.

They compare each metric to human hallucination labels (Table 1).

What they see:

GPT-4-as-judge is the most consistently aligned with humans across the different subsets.

Their ensemble (combine GPT-4, Critic, K-BERTScore, Q², consistency via FAMD) is a close second.

Critic is good on the dataset it was basically designed for (Begin, dialog-ish), but falls apart on HaluEval.

UniEval and some similarity metrics can look OK in numbers, but that’s partly because, in some subsets, they basically label almost everything as hallucinated — so they look “accurate” on a skewed dataset, but they’re not actually understanding hallucinations.

So the headline: outside GPT-4 (and ensembles), current metrics are spotty and dataset-sensitive.

Why this matters: if you compare two systems using a weak metric, you might think one hallucinates less — but that could just be the metric’s bias.

Finding 2: Metrics don’t agree with each other.

They compute inter-metric correlations and show: weak to no correlation.

Why? Because each metric is looking at a different slice of the hallucination problem:

Overlap metrics: “did you say the same words?”

Semantic metrics: “are you roughly talking about the same thing?”

Q²: “if I turn this into QA, does it still hold up?”

NLI-style (Critic): “does this contradict the source?”

GPT-4: “does this make sense in context, holistically?”

Those are not the same task. So two metrics can both be “good” on average but rarely fire on the same examples. Their Figure 2 shows this low overlap very explicitly.

Takeaway: hallucination isn’t 1-dimensional, so 1-dimensional metrics disagree. That’s also why their ensemble works better — it blends complementary signals.

Finding 3: Instruction-tuning and mode-seeking decoding reduce hallucinations.

They test whether hallucination scores change when:

you use an instruction-tuned version of a model vs base

you decode with greedy/beam (mode-seeking) vs sampling (top-k / top-p / ancestral)

They do significance tests and find:

Instruction-tuned models usually look less hallucinate-y to the better metrics (GPT-4, Critic) → so post-training helps, as prior papers suspected.

Greedy/beam tend to hallucinate less than sampling — consistent with earlier dialog work: sampling explores, and exploration can lead to making stuff up.

So: some levers we already use (tuning + decoding) do show up in hallucination metrics. That’s a nice confirmatory result.

Finding 4: Bigger models don’t automatically look “less hallucinated” to most metrics.

This one is the spicy one.

You’d expect: bigger model → better generations → hallucinate less → metric score improves.

But when they plot metric score vs model size, they don’t see a clean, monotonic improvement for most metrics.

Some metrics even get worse with size for some datasets.

Only GPT-4-as-judge shows a consistent “yeah, the bigger models look better” signal.

They also notice weird model behaviors (e.g. some Gemma models just abstain), which fool overlap-style metrics but are better picked up by GPT-4/Critic.

Their conclusion: if your metric doesn’t reflect scaling gains, maybe your metric isn’t really tracking hallucination.

That’s the “mirage” in the title: you think you’re measuring hallucination, but really you’re measuring something narrow like overlap, so you don’t see the benefits of better models.

4. What this means in practice

Here’s how I’d translate their conclusions for someone building/evaluating LLMs:

Don’t trust a single cheap metric (ROUGE, BERTScore, even some pretrained “consistency” models) to tell you “this system hallucinates less.” It might be dataset luck.

LLM-as-judge (GPT-4) is currently the strongest single metric for hallucination-style evaluation in their setup.

Ensembles help because hallucination has multiple faces (unsupported detail, wrong fact, off-topic, ungrounded).

Decode greedily/beam if you care about faithfulness — sampling will probably look worse.

Instruction-tune your model — evaluation metrics actually reflect less hallucination after tuning.

Be skeptical of papers that report small metric gains on one dataset — this paper shows lots of metrics don’t generalize across Begin ↔ HaluEval.

We still need better metrics — especially ones that are (a) robust across tasks, (b) sensitive to scaling, and (c) not easily gamed by always yelling “hallucination!”

5. Limitations they admit

They’re pretty honest:

They only have human labels on Begin and HaluEval, so the “alignment with humans” finding is only really grounded there.

They only study knowledge-grounded dialog and QA — not summarization, MT, code, etc., where hallucination looks a bit different.

They don’t cover uncertainty-based metrics that need token probs (semantic entropy, SAR) because those probs weren’t available in the public datasets.

They test GPT-4-as-judge, but not a whole zoo of judge variants (CoT judges, G-Eval, smaller judges).

So the message is “we checked a lot, but not everything.”

6. One-sentence takeaway

Most current hallucination metrics are brittle and don’t generalize; GPT-4-as-judge (or a good ensemble) is the most reliable right now, and you should be cautious about claiming “less hallucination” unless you’ve checked with something that actually tracks human judgment.
