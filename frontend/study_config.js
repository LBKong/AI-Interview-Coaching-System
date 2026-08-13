// Counterbalanced study plans. Questions stay in the same order while the RAG
// condition flips between groups, so each question appears in both conditions.
window.STUDY_PLANS = {
  A: [
    { question: "Tell me about a challenging project you worked on and your role in it.", rag: "on" },
    { question: "Describe a time you had a conflict with a teammate and how you handled it.", rag: "off" },
    { question: "Give an example of a time you led a team.", rag: "on" },
    { question: "Tell me about a time you failed or made a mistake, and what you did next.", rag: "off" },
  ],
  B: [
    { question: "Tell me about a challenging project you worked on and your role in it.", rag: "off" },
    { question: "Describe a time you had a conflict with a teammate and how you handled it.", rag: "on" },
    { question: "Give an example of a time you led a team.", rag: "off" },
    { question: "Tell me about a time you failed or made a mistake, and what you did next.", rag: "on" },
  ],
};
