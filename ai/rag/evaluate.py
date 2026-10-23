"""Evaluate FocusBot's RAG: retrieval quality, the "should I answer?" threshold, generation.

The question set below is hand-written for evaluation (not user data). Each question has
the knowledge section(s) that answer it, or None when the knowledge base cannot answer it.
Unanswerable FocusBuddy questions are the important ones: a model asked "how much does
premium cost?" will happily invent a price unless retrieval and the prompt stop it.

Usage:
    python -m ai.rag.evaluate                 # retrieval + threshold calibration (offline)
    python -m ai.rag.evaluate --generate      # also call the LLM (needs GROQ_API_KEY)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai.rag.retriever import KnowledgeRetriever
from nlp import retrieval_eval as rev

# (question, set of "doc#section-slug" that answer it, or None if unanswerable)
EVAL_QUESTIONS: list[tuple[str, set[str] | None]] = [
    ("How do I make a room private?", {"focusbuddy_rooms#public-and-private-rooms"}),
    ("Πώς φτιάχνω ιδιωτικό δωμάτιο;", {"focusbuddy_rooms#public-and-private-rooms"}),
    ("How long can a room name be?", {"focusbuddy_rooms#creating-a-room"}),
    ("Can the owner leave their own room?", {"focusbuddy_rooms#leaving-or-deleting-a-room"}),
    ("How do I remove someone from my room?", {"focusbuddy_rooms#managing-members"}),
    ("Who can pause the timer?", {"focusbuddy_timer#who-controls-the-timer"}),
    ("Ποιος μπορεί να ξεκινήσει το χρονόμετρο;", {"focusbuddy_timer#who-controls-the-timer"}),
    ("What is the longest focus session I can set?", {"focusbuddy_timer#focus-and-break-lengths"}),
    ("What happens when a focus block ends?", {"focusbuddy_timer#focus-and-break-cycles"}),
    ("Does reset save my session?", {"focusbuddy_timer#pausing-resetting-and-ending"}),
    ("How is my streak calculated?", {"focusbuddy_stats#streaks"}),
    ("Πώς βάζω ημερήσιο στόχο;", {"focusbuddy_stats#daily-goal", "focusbuddy_account#profile-settings"}),
    ("Can I download my focus history?", {"focusbuddy_stats#exporting-your-data"}),
    ("How many people are on the leaderboard?", {"focusbuddy_stats#leaderboard"}),
    ("I didn't get the verification email", {"focusbuddy_account#verifying-your-email"}),
    ("Ξέχασα τον κωδικό μου, τι κάνω;", {"focusbuddy_account#forgotten-password"}),
    ("What characters can my username have?", {"focusbuddy_account#registering"}),
    ("How do I switch to dark mode?", {"focusbuddy_account#appearance"}),
    ("How long can a chat message be?", {"focusbuddy_chat_tasks#room-chat"}),
    ("Someone is spamming the chat, can I stop them?", {"focusbuddy_chat_tasks#muting-members"}),
    ("Does the server see my screen share?", {"focusbuddy_chat_tasks#voice-and-screen-sharing"}),
    ("What is the Pomodoro technique?", {"study_techniques#the-pomodoro-technique"}),
    ("How should I review for an exam next month?", {"study_techniques#spaced-repetition",
                                                     "study_techniques#active-recall"}),
    ("Είναι καλό να διαβάζω όλη τη νύχτα πριν την εξέταση;", {"study_techniques#sleep-and-breaks"}),
    ("I keep checking my phone while studying", {"study_techniques#avoiding-multitasking"}),
    # Unanswerable from the knowledge base:
    ("How much does FocusBuddy premium cost?", None),
    ("Can I sync my sessions with Google Calendar?", None),
    ("Πώς αλλάζω τη γλώσσα της εφαρμογής;", None),
    ("Is there a mobile app for iPhone?", None),
    ("How do I delete my account?", None),
    ("What is the capital of France?", None),
    ("Recommend a good pizza place", None),
]


def section_key(chunk) -> str:
    from ai.rag.knowledge import _slug

    return f"{chunk.doc}#{_slug(chunk.section)}"


def retrieval_report(retriever: KnowledgeRetriever, k: int = 4) -> pd.DataFrame:
    """Per question: top similarity, rank of the first correct section, retrieval metrics."""
    rows = []
    for question, relevant in EVAL_QUESTIONS:
        scores = retriever.scores(question)
        order = np.argsort(-scores)
        ranked = list(dict.fromkeys(section_key(retriever.chunks[i]) for i in order))  # sections, deduplicated
        row = {"question": question, "answerable": relevant is not None, "top_similarity": float(scores[order[0]])}
        if relevant:
            row.update({"mrr": rev.reciprocal_rank(ranked, relevant),
                        f"hit@{k}": float(bool(set(ranked[:k]) & relevant)), "top_section": ranked[0]})
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_threshold(report: pd.DataFrame, grid: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    """Pick the min_similarity that best separates answerable from unanswerable questions by
    their top similarity (maximise balanced accuracy). Small labelled set: treat as a rough
    choice and re-check when the knowledge base changes."""
    grid = np.round(np.arange(0.10, 0.71, 0.025), 3) if grid is None else grid
    rows = []
    for t in grid:
        answers = report["top_similarity"] >= t
        tpr = (answers & report["answerable"]).sum() / report["answerable"].sum()
        tnr = (~answers & ~report["answerable"]).sum() / (~report["answerable"]).sum()
        rows.append({"threshold": float(t), "answer_rate_answerable": tpr, "refuse_rate_unanswerable": tnr,
                     "balanced_accuracy": (tpr + tnr) / 2})
    curve = pd.DataFrame(rows)
    best = curve["balanced_accuracy"].max()
    # Among ties, prefer the LOWEST threshold: refusing a good question is the costlier error here.
    chosen = float(curve.loc[curve["balanced_accuracy"] == best, "threshold"].min())
    return chosen, curve


REFUSAL_MARKERS = ("don't know", "do not know", "not sure", "couldn't find", "could not find", "no information",
                   "δεν ξέρω", "δεν γνωρίζω", "δεν βρήκα", "δεν έχω πληροφορ")


def generation_report(pipeline, questions=EVAL_QUESTIONS) -> pd.DataFrame:
    """Calls the LLM. Checks: answerable -> cites a valid source; unanswerable FocusBuddy
    questions -> says it does not know. A keyword check is crude; Phase 18 adds an
    LLM-as-judge."""
    rows = []
    for question, relevant in questions:
        result = pipeline.answer(question)
        text = result.answer.lower()
        rows.append({
            "question": question, "answerable": relevant is not None,
            "cites_source": bool(result.retrieved) and any(f"[{s['n']}]" in result.answer for s in result.sources),
            "says_unknown": any(m in text for m in REFUSAL_MARKERS),
            "invalid_citations": len(result.invalid_citations),
            "answer": result.answer[:160],
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from nlp.embeddings import SentenceTransformerEmbedder

    parser = argparse.ArgumentParser(description="Evaluate FocusBot retrieval and generation.")
    parser.add_argument("--generate", action="store_true", help="also query the LLM (needs GROQ_API_KEY)")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args(argv)

    retriever = KnowledgeRetriever(SentenceTransformerEmbedder())
    report = retrieval_report(retriever, args.k)
    answerable = report[report["answerable"]]
    print(f"Knowledge chunks: {len(retriever.chunks)}   questions: {len(report)} "
          f"({len(answerable)} answerable, {len(report) - len(answerable)} unanswerable)\n")
    hit = answerable[f"hit@{args.k}"].mean()
    print(f"Retrieval (answerable): MRR {answerable['mrr'].mean():.3f}, hit@{args.k} {hit:.3f}")
    misses = answerable[answerable[f"hit@{args.k}"] == 0]
    if len(misses):
        print("Missed:\n" + misses[["question", "top_section", "top_similarity"]].round(3).to_string(index=False))

    threshold, curve = calibrate_threshold(report)
    print("\nTop similarity: answerable vs unanswerable")
    print(report.groupby("answerable")["top_similarity"].describe()[["min", "25%", "50%", "max"]].round(3).to_string())
    print(f"\nCalibrated min_similarity: {threshold:.3f} (current default {retriever.min_similarity})")
    print(curve[(curve["threshold"] - threshold).abs() <= 0.1].round(3).to_string(index=False))

    if args.generate:
        from ai.llm.client import GroqLLM, LLMUnavailable
        from ai.rag.pipeline import RAGPipeline

        try:
            gen = generation_report(RAGPipeline(KnowledgeRetriever(retriever.embedder, min_similarity=threshold),
                                                GroqLLM()))
        except LLMUnavailable as exc:
            print(f"\nGeneration skipped: {exc}")
            return 0
        a, u = gen[gen["answerable"]], gen[~gen["answerable"]]
        print(f"\nGeneration: answerable citing a source {a['cites_source'].mean():.0%}, "
              f"unanswerable saying 'don't know' {u['says_unknown'].mean():.0%}, "
              f"invalid citations removed {gen['invalid_citations'].sum()}")
        print(gen[["question", "cites_source", "says_unknown", "answer"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
