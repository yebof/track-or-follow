---
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
  - typed-decision
  - structured-output
  - calibration
  - decision-making
  - deltanet
  - linear-attention
base_model: decider-2b
datasets:
  - limberc/this-that-spatial-bench
language:
  - en
---

# this-that-model-1.0

**A typed decision model. One forward pass, no decoding loop, no parser, no retry.**

Your program reaches a branch it cannot express in code — *is this refund within policy? is this
shell command safe to run unattended?* — and what it needs back is not prose. It is one of *n*
declared options and a number it can threshold.

This model returns exactly that, in about 31 ms on a consumer GPU, generating **zero** output
tokens.

- **Parameters:** 1.88 B
- **Architecture:** Qwen3.5-style hybrid, 18 of 24 layers DeltaNet linear attention, 6 full attention
- **Inference code:** [github.com/FLock-io/this-that-model](https://github.com/FLock-io/this-that-model)
- **Benchmark:** [limberc/this-that-spatial-bench](https://huggingface.co/datasets/limberc/this-that-spatial-bench)

## Usage

```bash
pip install git+https://github.com/FLock-io/this-that-model
```

```python
from thisthat import TypedDecider, Question

decider = TypedDecider.from_pretrained("flock-io/this-that-model-1.0")   # cuda, mps or cpu

answer = decider.decide(
    "command: rm -rf /var/lib/postgresql/data",
    Question("Is this shell command safe to run unattended on a production host?",
             ["yes, it only reads state",
              "no, it modifies or deletes data",
              "no, it contacts the network"]),
)

print(answer)                 # no, it modifies or deletes data (100%)
print(answer.index)           # 1
print(answer.probabilities)   # (0.001, 0.998, 0.001)

if answer.confidence < 0.8:
    escalate()                # the number is calibrated, so the threshold means something
```

Those outputs are measured, not illustrative. The escalation branch is not decoration: ask this
model something away from what it was trained on and it says so rather than guessing confidently.
`"user=alice tier=free requests_this_minute=847"` / *"Should this request be rate-limited?"*
returns **no (71%)** — unconfident, and for 847 requests a minute, wrong.

Several questions about one state are answered in the **same** forward pass, because no answer is
written back into the prompt and the decisions are conditionally independent given the input:

```python
answers = decider.decide(order_json, [
    Question("Within the refund window?", ["no", "yes"]),
    Question("Which queue?", ["standard", "priority", "manual review"]),
    Question("Risk band", ["low", "medium", "high"]),
])
```

On Apple Silicon, `from_pretrained` selects MPS automatically and loads in float16.

## Why a malformed answer is impossible

The answer is read from the hidden state at a designated position and scored against the label
tokens of the options you declared, normalised over exactly those:

$$p_k(j \mid x) = \operatorname{softmax}_j\left(\langle w_{\ell(k,j)}, h_k \rangle / \tau\right)$$

The support of that distribution **is** your option list. An answer outside it is not improbable,
it is unrepresentable. There is no token budget to exhaust, no letter to mis-parse, and no
`try/except` to write. Up to 255 options are supported; ten or fewer use an `(A) … (J)` rendering
and wider sets switch to single-token labels.

## Results

Every number below is reproducible from the inference repository; each script prints what it
measured beside what was published.

**Against `Jev`, a hosted commercial decision service, on 68 questions over 17 states that a third party
recorded and neither party chose.** The wording is theirs, not ours — this model was trained on a
different phrasing of the same question, so this measures transfer rather than recall.

![accuracy against the cost of one pass](accuracy-vs-cost.png)

| System | Accuracy | Brier ↓ | NLL ↓ | ms/question | Cost of one pass |
|---|---:|---:|---:|---:|---:|
| majority-class baseline | 0.647 | — | — | — | — |
| `claude-fable-5-1` | 0.676 | — | — | 2395 | $0.471 |
| `glm-5.3` | 0.721 | 0.204 | 0.601 | 819 | $0.008 |
| `qwen3.8-max` | 0.735 | 0.263 | 2.681 | 1014 | $0.021 |
| `NanoJev-0.6B` | 0.750 | 0.166 | 0.479 | — | — |
| **`Jev`** | 0.765 | 0.133 | 0.403 | — | — |
| `kimi-k3` | 0.779 | 0.143 | 0.430 | 989 | $0.009 |
| `deepseek-v4.1-flash` | 0.794 | — | — | 808 | $0.002 |
| `gpt-5.6` | **0.926** | — | — | 1180 | $0.018 |
| **this-that-model-1.0** | **0.941** | **0.042** | **0.126** | **30.9** | **$0.000014** |

A dash under Brier and NLL means the endpoint exposes no token probabilities, so those quantities
are not observable there — not that they are poor. Time is the median *per-question* latency at the
client. Our cost is electricity at 80 W and $0.30/kWh, a different kind of number from a price that
must cover serving and margin: read our distance from the hollow square as **one order of
magnitude, not five**. Jev's position on that axis is its *published* rate of $0.042 per million
input tokens applied to the token count we measured — its accuracy is ours to measure, its price is
theirs to state. Five further frontier models answered all 68 correctly and are omitted: at this
size they are saturated and rank nothing.

**Calibration, where the true answer is a computed probability.** A noisy actuator executes the
intended move with probability ρ and otherwise picks uniformly among the alternatives, so the
answer follows from the transition rules rather than from anyone's opinion.

| | Accuracy | qL2 ↓ |
|---|---|---|
| accuracy ceiling (computed — no predictor can exceed it) | 0.746 | — |
| constant predictor | — | 0.0962 |
| **this-that-model-1.0** | **0.750** | **0.0250** |

Of nine hosted frontier models measured on identical questions, only three expose token
probabilities at all, and **all three score worse than the constant predictor**.

**Decomposition.** The same questions about the same worlds, asked once from the whole rendered
map and once from the agent-centred window the answer provably depends on:

| World | Whole state | The part it depends on |
|---|---|---|
| 32×32 | 0.562 | 0.979 |
| 80×80 | 0.583 | 1.000 |
| 128×128 | 0.479 | 1.000 |
| 200×200 | 0.646 | 0.979 |

The whole-state input at 200×200 is 15,603 tokens; that rung was measured on an 80 GB card, the
others on a 16 GB laptop where it does not fit.

**On the released spatial benchmark**
([limberc/this-that-spatial-bench](https://huggingface.co/datasets/limberc/this-that-spatial-bench)),
7,305 questions over 15 families: **0.839** against a chance rate of 0.343, up from 0.409 before
this checkpoint was trained on those families. Read it with the caveat it deserves: the items and
the maze windows are held out by fingerprint *and* by rendered state, but the **question shapes
were trained on**, while every hosted system it is compared against met them for the first time at
test.

On the 2,250-question subset every system answered: `gpt-5.6` 0.897, `claude-opus-5` 0.892,
**this-that-model-1.0 0.844**, `claude-sonnet-5` 0.836, `Jev` 0.803, `glm-5.3` 0.789,
`laya-typed-decisions` 0.345, chance 0.343.

| system | accuracy | seen | unseen |
|---|---:|---:|---:|
| **this-that-model-1.0** | **0.844** | 0.873 | 0.834 |
| `laya-typed-decisions` | 0.345 | 0.430 | 0.314 |
| answering constantly | 0.343 | 0.425 | 0.314 |

The *seen* and *unseen* columns split the fifteen families by **our** training mixture, so for any
other system the gap between them reflects only that those four families are easier, not anything
about that system.
`Jev` is the closest comparison — the only other system there that generates no tokens — and that
figure was run for us by a third party, since we hold no key for it.

[Laya](https://github.com/NandhaKishorM/laya) is the closest published work by construction: a
non-autoregressive typed-decision engine over a 421M ModernBERT encoder, one forward pass, no
generated text. We ran `convaiinnovations/laya:typed-decisions` ourselves on this exact subset —
same items, same option sets, unanswered counted as wrong — and it scored **0.345 against a chance
rate of 0.343**, with one family of fifteen clearing chance by more than a standard error.

That is zero-shot on a task family it was never trained for, which its authors put first: *"Laya
is a fast base to specialise, not a zero-shot decision engine."* The number worth reading beside it
is our own: **this checkpoint scored 0.409 here before it was trained on these families, which is
also chance.** What separates the two rows is training data, not architecture. And Laya answered in
a median of **24 ms** against our 30.9, on a model four and a half times smaller — on the axis a
zero-shot comparison can actually settle, it is ahead.

| the state rendered as | before | after | chance |
|---|---|---|---|
| an ASCII block | 0.457 | 0.779 | 0.388 |
| the same cells as JSON | **0.373 (chance)** | **0.936** | 0.394 |
| the same cells as prose | 0.420 | 0.942 | 0.394 |

## Training

Adapted from `decider-2b` (Apache-2.0) against a strictly proper scoring rule, so the model has no
way to lower its loss except by reporting what it believes — which is what makes the probability
usable as a threshold. The whole adaptation is a single scalar λ applied as θ(λ) = θ₀ + λΔ, so
θ(0) is bit-exact the prior checkpoint and rollback is a configuration change rather than a
restore.

## Citation

```bibtex
@misc{cheng2026thisthat,
  title  = {A typed decision model that decides in 30 ms, for a millionth of a cent},
  author = {Cheng, Zehua and Dai, Wei and Sun, Jiahao},
  year   = {2026},
  note   = {University of Oxford and FLock.io}
}
```

## Licence and attribution

MIT. Adapted from `decider-2b` under Apache-2.0. The environment simulator and the recorded
68-question cohort come from [NanoJev](https://github.com/TianyuCodings/NanoJev) under the MIT
licence and are used as published; we thank its authors.
