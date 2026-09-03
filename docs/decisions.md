# Decisions

Short notes on the choices behind this harness, and what each one costs.

## The dataset was chosen for its ground truth, not its domain

The point of the repo is the measurement, which inverts the usual corpus
question. Scientific abstracts are not inherently interesting; what matters is
that SciFact carries human judgments at both stages: 300 test queries with
relevance judgments, and 188 of those with a human SUPPORT or CONTRADICT label.

Both halves are needed. Retrieval judgments alone would leave answer quality to
an unvalidated judge. Having the verdict labels too is what makes it possible to
measure the judge before trusting it.

It also has a published baseline. BM25 on SciFact scores about 0.665 nDCG@10 in
the BEIR paper, and this implementation scores 0.6622. A hand written scorer
that had landed at 0.45 would look perfectly plausible with nothing to check it
against.

Cost: the documents are short, median 204 words, so chunk size ablations have
nothing to vary. That is stated in the README rather than quietly omitted.

## Metrics are tested against values worked out by hand

Testing a metric against its own output only proves it is self consistent. The
tests assert that a relevant document in second place scores exactly 1/log2(3),
and so on. If the implementation and the test are both wrong in the same way,
the suite is decoration.

## Everything comes with a confidence interval

The test split is 300 queries. A one point difference between configurations is
often noise, and reporting the mean alone is how a portfolio ends up claiming a
gain that is not there. Every number carries a bootstrap interval, and the
ablation marks a comparison as inconclusive rather than picking the larger mean.

What decides that marking is the interval of the difference, not whether the two
intervals overlap. Both configurations are scored on the same queries, so the
difference can be bootstrapped per query. Reading two separate intervals throws
that pairing away: they can overlap while every single query moved the same way,
and a real improvement gets called a tie.

## Configuration is chosen on train, reported on test

The 809 training queries exist so that choices, RRF constant, top-K, embedding
model, can be made without touching the 300 test queries. Selecting a
configuration by looking at the test numbers is the same mistake as calibrating
a threshold by looking at the answer.

## Fusion works on positions, not scores

BM25 returns tens and cosine returns fractions. Weighted sum fusion has to
normalise two different distributions to compare them, and that normalisation is
usually where hybrid retrieval loses to whichever component it was meant to
improve. Reciprocal rank fusion only needs the ordering.

The constant is configurable. 60 is what the literature uses, which is not the
same as a measurement on this corpus.

## The judge is measured before it is used

Judge scores are everywhere and validated judges are not. An LLM judge favours
long answers and its own style, and agrees too readily when the rubric is vague.
An unvalidated score is an opinion with decimal places.

So the judge answers the same 188 claims that have human labels, and its
agreement is reported as accuracy and Cohen's kappa. Kappa because accuracy
flatters a classifier on unbalanced classes, and SUPPORT outnumbers CONTRADICT
close to two to one here: a judge that answered SUPPORT every time would score
high accuracy and zero kappa.

Only then is it used for what has no ground truth. Its agreement is printed next
to its scores, never in a separate section. If the agreement turns out poor,
that is the result, and it goes in the README as it stands.

## Nothing is silently coerced

Three places where the easy thing would corrupt the measurement:

- A verdict outside the label set is refused, not mapped to NOINFO. Mapping it
  would make the accuracy figure describe a fallback rather than the model.
- A citation pointing outside the passages is counted, not dropped. Dropping it
  would erase the defect the faithfulness metric exists to show.
- A refusal from the API is raised, not treated as a wrong answer. It arrives as
  a normal 200 and would quietly depress the score.

## The gate treats its two halves differently

Retrieval is deterministic: no model is involved, so any drop is a real change
and the tolerance is zero.

Generation is not. Sampling controls were removed from the current models, so
there is no temperature to pin at zero and two identical runs disagree. A
tolerance band picked by feel fails on noise, everyone learns to rerun until it
passes, and from then on the gate protects nothing. So generation reports
without blocking until the run to run variance has actually been measured, and
the band comes from that measurement.

The gate also refuses to compare a run against a baseline recorded on a
different model. Running something cheap day to day and measuring it against an
expensive baseline is the most tempting way to get a wrong answer here.

## Four providers behind one interface

A deterministic stub for the suite and CI, so tests need neither network nor
API key and never go flaky. Ollama so a clone runs for free. Anthropic and
Gemini for the numbers that get published.

Every reported number carries which provider and model produced it. Comparing an
Ollama run to an Opus run as though they were the same measurement would be
exactly the sort of wrong number this repo is about.

The published numbers here come from Gemini on the free tier, which shapes the
run in two ways worth stating rather than hiding. The quota is 15 requests per
minute per model, so a pass over the 188 labelled claims takes about thirteen
minutes and no amount of concurrency changes that; the client paces itself
instead of earning a 429 with a minute long delay attached. And free tier
traffic may be used to improve the provider's models, which is acceptable here
only because every prompt is built from a public benchmark.

The subscription route through the Claude CLI is deliberately not offered here.
A public repo invites strangers to run it, and the terms do not cover using a
subscription as a product backend.

## The judge does not grade its own work

The verifier and the judge run on different models by default. A model asked to
grade its own answer rates it generously, and an agreement figure produced that
way measures a preference rather than a capability.

Which second model is not a free choice on this tier. The quota is per model and
it is not published: gemini-3-flash-preview turned out to allow twenty requests
a day, which a run of 188 claims discovers about eighty percent of the way
through. The judge runs on Gemma, which took a burst of twenty two without
complaint, and the agreement figure that follows says what that choice was
worth.
