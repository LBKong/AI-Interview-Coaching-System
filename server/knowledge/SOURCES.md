# Knowledge Base — Sources

The RAG knowledge base is written in our own words from the reputable sources listed below.
Every entry in `general.json` and `questions.json` carries a `source` field recording its
provenance. No entry is AI-generated content presented as coaching guidance, and no text is
copied verbatim from the sources.

## Academic literature

**Martín-Raugh, M. P., et al. (2023).** Speaking without words: A meta-analysis of over 70 years
of research on the power of nonverbal cues in job interviews. *Journal of Organizational
Behavior*, 44, 132–156.
→ Backs the general principle on nonverbal delivery, and provides the empirical justification
for the two delivery metrics the system computes (gaze / eye contact and speaking rate).

**Levashina, J., Hartwell, C. J., Morgeson, F. P., & Campion, M. A. (2014).** The Structured
Employment Interview: Narrative and Quantitative Review of the Research Literature. *Personnel
Psychology*.
→ Backs principles on relevance to the assessed competency, impression management, and the
behavioural-interview premise (actions rather than feelings).

**Campion, M. A., Palmer, D. K., & Campion, J. E. (1997).** A review of structure in the
selection interview. *Personnel Psychology*, 50(3), 655–702.
→ Backs the account of how structured interviews score answers against predetermined criteria.

**Chollet, M., Marsella, S., & Scherer, S. (2022).** Training public speaking with virtual social
interactions: effectiveness of real-time feedback and delayed feedback. *Journal on Multimodal
User Interfaces*, 16, 17–29.
→ Supports the delayed (post-session) feedback design, and the use of objective performance
measures such as eye contact and pause fillers in a follow-up report.

**Fourati, N., Barkar, A., Dragée, M., Danthon-Lefebvre, L., & Chollet, M. (2025).** Probing
Experts' Perspectives on AI-Assisted Public Speaking Training. arXiv:2507.07930.
→ Backs the principles that a delivery measure means little unless linked to the higher-level
quality it signals, and that there is no universal standard for speaking rate. Also informs the
feedback-design principles applied in the prompt (see note below).

## Practitioner guidance

**UK National Careers Service** — interview advice and the STAR method.
`nationalcareers.service.gov.uk/careers-advice/interview-advice`
→ Backs the STAR structure and general interview-preparation principles.

**Prospects.ac.uk** — competency-based interviews and interview questions.
`prospects.ac.uk/careers-advice/interview-tips`
→ Backs the per-question guidance: which competency each question targets, the question set
itself, and the common pitfalls for each.

**University careers-service guidance on STAR** (Yale, Northwestern, University of
Wisconsin–Madison).
→ Backs the detailed STAR guidance: Action as the longest section, using "I" rather than "we",
specificity over generalisation, and giving enough detail without rambling.

## Note on the division between knowledge base and prompt

The knowledge base holds **domain knowledge** — what a given question assesses, what a strong
answer looks like, common pitfalls, and research findings on delivery. **Feedback-design
principles** (how many points to raise, balancing strengths with improvements, tone) are applied
in the prompt instead, so that they stay constant when the RAG switch is turned off. This keeps
the ablation comparing knowledge, not feedback style.
