# Feedback Report Structure — Design Proposal

**Scope:** the structure of one feedback report on one interview answer (L1, post-session text).
**Purpose:** settle *what goes in the report and what stays out* before `feedback.py` is written,
since the prompt has to produce this structure.

---

## Where the design comes from

The supervisor asked for the report design to be grounded in literature and existing systems
rather than invented. Two sources do most of the work here.

**Existing systems** (Fourati et al., 2025, Tables 1 and 3):

| System | What it reports |
|---|---|
| **MACH** — simulated interview trainer, post-session feedback | speech tempo, fillers, prosody, facial cues. *The closest predecessor to this project: an interview trainer with delayed feedback, evaluated with 90 users, improved interview skills and self-presentation.* |
| **Cicero (2022)** — virtual audience, after-action report | graphs and comments on facial expressions, pauses, gaze. *Improved confidence, eye contact and flow of speech **only when the after-action report contained personalised feedback**.* |
| **Microsoft Speaker Coach** | filler words, pace, intonation, repetitions |
| **Yoodli** | words per minute, pauses, eye-contact duration, filler words; AI coaching on strengths and growth areas; per-response breakdown in roleplays |
| **Orai** | scores for Energy, Conciseness, Confidence, Pace |
| **Poised** | goal-based feedback highlighting strengths and areas to improve, with examples of the user's own phrasing |
| **VocaCoach** | 1–5 scores on five vocal criteria, spider chart |

**Expert criticism of those systems** (same paper, Insights 2 and 3, and the Conclusion) — this is
what shapes the design more than the feature lists:

- Feedback must be **carefully selected and limited to a few key elements**, organised logically
  and hierarchically. One coach: *"I give 2-3 comments maximum on trainees' performances,
  otherwise they will not remember anything and it will be useless."*
- The over-detailed system (PolymnIA) was criticised for **cognitive load** — users could not tell
  what to focus on — and for mixing high-level and low-level criteria side by side with no order
  of importance.
- The over-simple system (VocaCoach) was criticised for being **positively biased**, which risks
  inflating a learner's confidence misleadingly.
- **Low-level measurements must be linked to the higher-level quality they signal**, otherwise
  they are not actionable and push learners towards a uniform, inauthentic style. *"Speaking speed
  doesn't mean much in itself."*
- The conclusion asks for both **positive reinforcement and areas of improvement**, and for
  bridging behavioural cues with high-level communication criteria.

---

## Proposed structure — five short sections

Fixed order, because the ordering *is* the hierarchy the experts asked for: the judgement that
matters most comes first, and the behavioural detail comes last.

### 1. Did the answer address the question? — always present, always first

One or two sentences giving a direct verdict on relevance to the competency the question targets.

*Why first:* in a structured interview the answer is scored against criteria set in advance for
that competency, so a fluent answer that misses the competency still scores poorly. This is also
the accuracy-first check already in the rubric. Putting it first prevents the report from reading
as a list of equally-weighted observations.

### 2. What worked — 1–2 specific strengths

Must refer to something the candidate actually said, not generic praise.

*Why included:* automated systems tend to focus on weaknesses while expert coaches build on
strengths; the paper's conclusion asks for both. *Why constrained:* generic praise is what made
VocaCoach's positive bias a problem.

**Honesty override — this section is not mandatory.** If the answer has no genuine strength, the
report must not manufacture one. It may acknowledge something real but minor (a suitable choice
of example, a willingness to be candid) without endorsing the answer's quality, or it may omit
the section entirely. Inventing a strength to fill the slot is the single fastest way to score 1
on the rubric's Accuracy dimension, which explicitly penalises calling a weak answer "solid"
however positive it sounds.

### 3. What to improve — 1–2 concrete, actionable points

Each point says what to do differently, not just what was wrong.

### 4. Delivery — present whenever the signals exist; omitted only when they do not

**When the multimodal signals are available, this section always appears** — either flagging
something notable, or stating briefly that nothing stood out. **When the signals are switched off
it is omitted entirely**, because there is nothing to report.

The earlier draft of this design omitted the section whenever nothing was notable. That was
changed after checking against the rubric: Coverage asks whether the aspects that should be
addressed — content *and* delivery — were addressed, and a silent omission is indistinguishable
to a rater from delivery having been ignored. A one-line "nothing stood out" costs almost nothing
and makes the judgement visible.

Every number must be attached to what it may communicate. "You spoke at 185 words per minute" alone is not acceptable output;
"185 words per minute, on the quick side, which together with limited eye contact can read as
discomfort" is.

*Why conditional and last:* nonverbal cues do influence interviewer judgements even after content
is accounted for (Martín-Raugh et al., 2023), so they belong in the report — but content comes
first, and an isolated metric is not actionable.

### 5. One thing to try next time — a single sentence

A restatement of the single highest-priority action, not a new point.

*Why:* Cicero's after-action report only produced improvement when the feedback was personalised;
a concrete, specific closing action is what makes the report personal rather than generic.

### Hard cap on substance

**At most three substantive points total across sections 2 and 3.** Normally at least one of
each, except where the honesty override applies and there is no genuine strength to report.
Section 5 restates one of them rather than adding a fourth. This is the "2-3 comments maximum"
constraint applied directly.

**Target length: about 150–220 words.** Note this is a constraint on the *report*, and is separate
from the rubric's length-neutrality requirement, which governs how the *answer* is judged.

---

## Deliberately excluded, and why

| Excluded | Reason |
|---|---|
| **Any overall numeric score or rating of the candidate** | Opaque aggregation was a central criticism of the commercial tools. Methodologically it is also a trap here: if the system prints a score, participants may end up rating the score rather than the feedback, and expert raters would be scoring a number they cannot verify. The rubric scores the *feedback*; the system should not score the *candidate*. |
| **A raw metric dashboard** (a table of WPM, gaze %) | Isolated metrics without interpretation were the main complaint about the over-detailed system. Numbers appear only inside section 4, attached to a meaning. |
| **Percentile or peer comparison** | Not available with this sample, and it pushes towards conformity — the exact risk the experts raised. |
| **Comments on transcription artifacts** (punctuation, ASR errors) | Not the candidate's behaviour. |
| **Filler words / hesitations** | Reported by almost every existing system, and the supervisor mentioned pauses as easy to compute — but the system does not currently measure them. Listed here as the obvious first extension, not as part of the current design. |

---

## Keeping the ablation clean

**The structure stays identical in all four conditions.** RAG on/off and multimodal on/off change
what the report can *say*, never its shape:

- **RAG off** → every section that would have appeared still appears, in the same order and at the
  same length. Only the grounding of the content changes: the points come from the model's own
  knowledge rather than from the retrieved sources. **The two RAG conditions must be
  indistinguishable by shape.**
- **Multimodal off** → section 4 disappears, because the signals genuinely do not exist.

The asymmetry is deliberate. RQ2a compares RAG on against RAG off, and the participant
questionnaire measures trust and perceived usefulness — so if the RAG-off reports also *looked*
different, participants would be reacting to a format change rather than to the grounding, and
RQ2a would be confounded. RQ2b is a different case: the multimodal condition really does have
more information, and a visibly shorter report is an honest consequence of that, not an artifact.
What must not happen there is the rubric treating that honest absence as a failure — see Conflict
2 below.

---

## Worked example

Question: *Describe a time you had a conflict with a teammate and how you handled it.*
Answer: *"One teammate wasn't doing their part. It was frustrating. I ended up just doing most of
the work myself so we could finish on time."*
Signals: 185 wpm, 40% on-camera. RAG on, multimodal on.

> **Did you answer the question?**
> Not quite. The question asks how you handled the conflict, but the answer describes how you
> worked around it — by taking the work on yourself, the disagreement with your teammate was never
> actually addressed.
>
> **What worked**
> You chose a real situation and were honest about how it ended, rather than presenting it as a
> success. Interviewers notice that.
>
> **What to improve**
> Show the interviewer what you said or did with your teammate — a conversation you started, a
> compromise you proposed. That handling is the thing being assessed. Also, "it was frustrating"
> tells them how it felt; what they are listening for is what you decided to do about it.
>
> **Delivery**
> You spoke at about 185 words per minute, on the quick side, and looked at the camera around 40%
> of the time. Together those can read as discomfort with the topic — worth knowing, since
> conflict questions often produce exactly that.
>
> **Next time**
> Keep the same story, but spend most of the answer on the conversation you had with your
> teammate.

*(181 words. Three substantive points: one strength, two improvements.)*

With multimodal off, the Delivery section is absent and the rest is unchanged. With RAG off, the
same sections appear but the improvement points are not grounded in the retrieved pitfall about
avoiding rather than handling conflict.

---

## Alignment with rubric v2

Every rubric dimension has something in the report that can satisfy it, and no report section is
left unevaluated.

| Rubric dimension | Where the report produces it |
|---|---|
| **Accuracy** — is the judgement of the answer correct? | Section 1 carries most of it. Placing it first mirrors the rubric's own accuracy-first check. |
| **Specificity** — tied to this answer, not generic | Sections 2 and 3 must refer to what was actually said; section 4 uses the actual measured values. |
| **Actionability** — can the candidate act on it? | Section 3 states what to do differently; section 5 exists solely to leave one concrete next step. |
| **Coverage** — content *and* delivery addressed | Sections 1–3 cover content; section 4 covers delivery. |
| **Overall usefulness** | Emergent from the other four; no dedicated section. |

Checking the two sets against each other surfaced two conflicts, both now fixed above.

**Conflict 1 — a mandatory strengths section fights the Accuracy dimension.** Requiring at least
one strength in every report means that on a genuinely weak answer the system must either invent
one or stretch. The rubric penalises exactly that: a weak answer described as "solid" scores 1
on Accuracy however positive the wording. Resolved by the honesty override in section 2.

**Conflict 2 — Coverage would have penalised the multimodal-off condition for something it
cannot do.** Coverage asks whether content *and* delivery were addressed. With the multimodal
signals switched off the report structurally cannot address delivery, so it would score lower on
Coverage — and RQ2b would then partly measure the rubric penalising a condition for a missing
capability, rather than measuring whether nonverbal signals improve feedback. Two changes are
needed, one in the report and one in the rubric:

- *Report side (done):* when the signals exist, section 4 always appears, so a rater can see
  delivery was considered rather than having to infer it from an absence.
- *Rubric side (needs a decision):* Coverage should be scored **relative to the information
  available to the system** — did the report address the aspects it had signals for — rather than
  against a fixed content-plus-delivery checklist. Otherwise the multimodal ablation is
  confounded. This wording change belongs in the rubric and in the rater instructions, and is
  worth raising with the supervisor since he proposed the trust-and-usefulness measures that sit
  alongside it.
