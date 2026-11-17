"""Evaluation harness: runs the RAG advisor on predefined cases and prints a pass/fail summary.

Run:  python eval_harness.py
Exit code is 1 if any case fails, so it can be used in CI.

Each case states what a GOOD answer looks like:
  status           grounded / fallback / urgent
  cite             a substring that must appear in at least one citation
  forbid           a substring that must NOT appear in any citation (e.g. another pet's profile)
  warns            True if a duration warning is expected
  reminder         True if the "never recommends doses" reminder must be present
"""

from __future__ import annotations

import sys
from datetime import datetime

from advisor import GuidanceAdvisor
from pawpal_system import Category, Frequency, Pet, Priority, Task
from retriever import KnowledgeRetriever, configure_logging

WHEN = datetime(2026, 11, 17, 9, 0)

CASES = [
    # name, pet, task(description, category, minutes), expectations
    ("Chester 40-min walk is flagged",
     Pet("Chester", "dog", 12, notes="yorkie schnauzer mix"), ("Long walk", Category.WALK, 40),
     dict(status="grounded", cite="chester_profile.md#Chester's walks", warns=True)),
    ("Chester 15-min walk is not flagged",
     Pet("Chester", "dog", 12, notes="yorkie schnauzer mix"), ("Short walk", Category.WALK, 15),
     dict(status="grounded", cite="chester_profile.md#Chester's walks", warns=False)),
    ("Chester breakfast uses his feeding notes",
     Pet("Chester", "dog", 12, notes="yorkie schnauzer mix"), ("Breakfast", Category.FEEDING, 10),
     dict(status="grounded", cite="Chester's feeding")),
    ("Chester pill carries the vet reminder",
     Pet("Chester", "dog", 12, notes="yorkie schnauzer mix"), ("Morning pill", Category.MEDICATION, 5),
     dict(status="grounded", reminder=True)),
    ("Chester tooth brushing uses dental guidance",
     Pet("Chester", "dog", 12, notes="yorkie schnauzer mix"), ("Brush teeth", Category.GROOMING, 5),
     dict(status="grounded", cite="teeth")),
    ("Other senior dog gets general senior-walk advice, not Chester's profile",
     Pet("Rex", "dog", 11, notes="terrier"), ("Evening walk", Category.WALK, 30),
     dict(status="grounded", cite="dog_exercise_by_age_and_size.md", forbid="chester_profile", warns=True)),
    ("Puppy feeding uses puppy guidance",
     Pet("Pip", "dog", 0), ("Puppy meal", Category.FEEDING, 10),
     dict(status="grounded", cite="dog_feeding_schedule.md", forbid="senior")),
    ("Cat feeding uses cat guidance, not dog",
     Pet("Luna", "cat", 4, notes="indoor"), ("Dinner", Category.FEEDING, 5),
     dict(status="grounded", cite="cat_care_basics.md", forbid="dog")),
    ("Hamster has no guidance -> honest fallback",
     Pet("Nibbles", "hamster", 1), ("Refresh bedding", Category.GROOMING, 10),
     dict(status="fallback")),
    ("Hamster pill: general medication-safety doc applies to any species, reminder kept",
     Pet("Nibbles", "hamster", 1), ("Morning pill", Category.MEDICATION, 5),
     dict(status="grounded", cite="medication_safety.md", reminder=True)),
    ("Emergency wording is escalated",
     Pet("Chester", "dog", 12, notes="yorkie schnauzer mix"), ("Chester had a seizure", Category.APPOINTMENT, 30),
     dict(status="urgent")),
    ("Gibberish description does not invent guidance",
     Pet("Rex", "dog", 5), ("qzx vwv", Category.ENRICHMENT, 10),
     dict(status="grounded_or_fallback")),
]


def run_case(advisor: GuidanceAdvisor, pet: Pet, spec: tuple, expect: dict) -> tuple[bool, str, float]:
    description, category, minutes = spec
    task = Task(description, WHEN, minutes, category, Priority.MEDIUM, Frequency.ONCE)
    advice = advisor.advise_task(pet, task)
    problems = []

    wanted = expect["status"]
    if wanted == "grounded_or_fallback":
        if advice.status == "grounded" and advice.confidence < 0.3:
            problems.append(f"low-confidence match ({advice.confidence}) was presented as grounded")
    elif advice.status != wanted:
        problems.append(f"status {advice.status!r}, expected {wanted!r}")
    if "cite" in expect and not any(expect["cite"].lower() in c.lower() for c in advice.citations):
        problems.append(f"no citation containing {expect['cite']!r} (got {advice.citations})")
    if "forbid" in expect and any(expect["forbid"].lower() in c.lower() for c in advice.citations):
        problems.append(f"forbidden source {expect['forbid']!r} was cited")
    if "warns" in expect and bool(advice.warnings) != expect["warns"]:
        problems.append(f"warning present={bool(advice.warnings)}, expected {expect['warns']}")
    if expect.get("reminder") and "never recommends doses" not in advice.message:
        problems.append("medication reminder missing")
    return (not problems, "; ".join(problems) or "ok", advice.confidence)


def main() -> int:
    configure_logging("logs/eval.log")
    advisor = GuidanceAdvisor(KnowledgeRetriever())
    print(f"PawPal+ RAG evaluation  ({len(advisor.retriever.chunks)} knowledge chunks, "
          f"threshold {advisor.threshold})")
    print("-" * 96)
    print(f"{'#':>2}  {'result':<6} {'conf':>5}  case")
    print("-" * 96)
    passed = 0
    grounded_scores = []
    for number, (name, pet, spec, expect) in enumerate(CASES, start=1):
        ok, detail, confidence = run_case(advisor, pet, spec, expect)
        passed += ok
        if confidence and expect["status"] == "grounded":
            grounded_scores.append(confidence)
        print(f"{number:>2}  {'PASS' if ok else 'FAIL':<6} {confidence:>5.2f}  {name}")
        if not ok:
            print(f"                 -> {detail}")
    print("-" * 96)
    average = sum(grounded_scores) / len(grounded_scores) if grounded_scores else 0.0
    print(f"{passed}/{len(CASES)} cases passed; average confidence on grounded answers: {average:.2f}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
