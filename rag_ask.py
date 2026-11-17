"""Ask the guidance layer about one task, from the command line.

Examples:
    python rag_ask.py --pet Chester --species dog --age 12 --notes "yorkie schnauzer mix" \\
        --task "Long walk" --category walk --minutes 40
    python rag_ask.py --pet Nibbles --species hamster --age 1 --task "Refresh bedding" --category grooming
    python rag_ask.py --pet Chester --species dog --age 12 --task "Chester had a seizure" --category appointment
"""

from __future__ import annotations

import argparse
from datetime import datetime

from advisor import GuidanceAdvisor, life_stage
from pawpal_system import Category, Frequency, Pet, Priority, Task
from retriever import KnowledgeRetriever, configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pet", required=True)
    parser.add_argument("--species", required=True, help="dog, cat, or anything else")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--notes", default="", help="breed, conditions, habits")
    parser.add_argument("--task", required=True, help="what you plan to do, e.g. 'Long walk'")
    parser.add_argument("--category", required=True, choices=[c.value for c in Category])
    parser.add_argument("--minutes", type=int, default=15)
    args = parser.parse_args()

    configure_logging("logs/pawpal_rag.log")
    pet = Pet(args.pet, args.species.lower(), args.age, notes=args.notes)
    task = Task(args.task, datetime.now(), args.minutes, Category(args.category), Priority.MEDIUM,
                Frequency.ONCE)
    advice = GuidanceAdvisor(KnowledgeRetriever()).advise_task(pet, task)

    print(f"Pet:    {pet.name} ({pet.species}, {pet.age} yr, {life_stage(pet.species, pet.age)})")
    print(f"Task:   {task.description} [{task.category.value}, {task.duration_minutes} min]")
    print(f"Status: {advice.status} (confidence {advice.confidence:.2f})")
    print(f"Advice: {advice.message}")
    for warning in advice.warnings:
        print(f"Warning: {warning}")
    if advice.citations:
        print(f"Sources: {', '.join(advice.citations)}")


if __name__ == "__main__":
    main()
