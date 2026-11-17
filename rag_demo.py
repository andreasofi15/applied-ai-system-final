"""CLI demo: PawPal+ daily plan grounded in the knowledge base (RAG).

Run:
    python rag_demo.py              # show the plan and the pending suggestions
    python rag_demo.py --accept     # also approve every suggestion and re-print the plan
    python rag_demo.py --ask        # approve suggestions one by one (human in the loop)
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from advisor import GuidanceAdvisor
from pawpal_system import (Category, Frequency, Owner, Pet, Priority, Scheduler, Task,
                           TimeBlock, TimePreference)
from retriever import KnowledgeRetriever, configure_logging

DAY = date(2026, 11, 17)


def at(hour: int, minute: int = 0) -> datetime:
    return datetime.combine(DAY, datetime.min.time()).replace(hour=hour, minute=minute)


def build_owner() -> Owner:
    from datetime import time
    owner = Owner("Andrea", available_minutes=150)
    owner.add_busy_block(TimeBlock("Class", ["Tue"], time(13, 0), time(15, 0)))

    chester = Pet("Chester", "dog", 12, notes="yorkie schnauzer mix")
    chester.add_task(Task("Breakfast", at(8), 10, Category.FEEDING, Priority.HIGH,
                          Frequency.DAILY, TimePreference.MORNING, must_do=True))
    chester.add_task(Task("Morning pill", at(8, 15), 5, Category.MEDICATION, Priority.HIGH,
                          Frequency.DAILY, TimePreference.MORNING, must_do=True))
    chester.add_task(Task("Long walk around the park", at(9), 40, Category.WALK, Priority.MEDIUM,
                          Frequency.DAILY, TimePreference.MORNING))
    chester.add_task(Task("Brush coat", at(18), 10, Category.GROOMING, Priority.LOW, Frequency.WEEKLY))

    luna = Pet("Luna", "cat", 4, notes="indoor")
    luna.add_task(Task("Dinner", at(18, 30), 5, Category.FEEDING, Priority.HIGH,
                       Frequency.DAILY, TimePreference.EVENING, must_do=True))
    luna.add_task(Task("Wand toy play", at(19), 10, Category.ENRICHMENT, Priority.MEDIUM))

    nibbles = Pet("Nibbles", "hamster", 1, notes="")
    nibbles.add_task(Task("Refresh bedding", at(20), 10, Category.GROOMING, Priority.LOW))

    for pet in (chester, luna, nibbles):
        owner.add_pet(pet)
    return owner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accept", action="store_true", help="approve all suggestions")
    parser.add_argument("--ask", action="store_true", help="approve suggestions interactively")
    args = parser.parse_args()

    configure_logging("logs/pawpal_rag.log")
    owner = build_owner()
    advisor = GuidanceAdvisor(KnowledgeRetriever())
    scheduler = Scheduler(owner, advisor)

    print("=" * 70)
    print("BASELINE (no retrieval): reasons come only from priority and time")
    print("=" * 70)
    print(Scheduler(owner).build_daily_plan(DAY).summary())

    print()
    print("=" * 70)
    print("WITH RAG: reasons and warnings come from retrieved, cited guidance")
    print("=" * 70)
    plan = scheduler.build_daily_plan(DAY)
    print(plan.summary())

    approved = []
    for suggestion in plan.suggestions:
        if args.accept:
            approved.append(suggestion)
        elif args.ask:
            answer = input(f"\nAdd '{suggestion.description}' for {suggestion.pet_name}? [y/N] ")
            if answer.strip().lower().startswith("y"):
                approved.append(suggestion)

    if approved:
        print()
        print("=" * 70)
        print(f"AFTER APPROVING {len(approved)} SUGGESTION(S): added to conflict-free time slots")
        print("=" * 70)
        for suggestion in approved:
            task = scheduler.accept_suggestion(suggestion, DAY)
            print(f"  + {task.pet_name}: {task.description} at {task.due_time:%H:%M}")
        print()
        print(scheduler.build_daily_plan(DAY).summary())


if __name__ == "__main__":
    main()
