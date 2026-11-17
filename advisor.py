"""GuidanceAdvisor: turns retrieved knowledge into advice the Scheduler can use.

Pipeline for each task:
    pet profile + task -> query -> retriever -> confidence check -> advice (with citations)

Guardrails:
  * Emergency wording in a task ("seizure", "poison", ...) skips scheduling advice and says to call a vet.
  * Low retrieval confidence returns an honest "ask your vet" fallback instead of a guess.
  * Medication tasks always carry a "follow your vet's directions" reminder; the system never states doses.
  * Anything that goes wrong inside retrieval is logged and degrades to the fallback, never a crash.

The advisor is duck-typed: it only reads pet.name/species/age/notes and task.category/description/
duration_minutes, so it does not import pawpal_system (no circular imports).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from retriever import Hit, KnowledgeRetriever, Suggestion, tokenize

logger = logging.getLogger("pawpal.rag")

DEFAULT_THRESHOLD = 0.15          # below this top score, we do not trust the match
MEDICATION_REMINDER = "Always follow your vet's directions; PawPal+ never recommends doses."
FALLBACK_MESSAGE = "No reliable guidance found for this task. Ask your vet."
CATEGORY_TERMS = {
    "feeding": "meal food eat",
    "walk": "walk exercise",
    "medication": "medication pill",
    "grooming": "grooming brush coat",
    "enrichment": "enrichment play toy",
    "appointment": "vet checkup visit",
}
URGENT_TERMS = (
    "seizure", "collapse", "poison", "toxic", "trouble breathing", "can't breathe",
    "cannot breathe", "bloody", "blood in", "swollen belly", "choking", "unresponsive",
)


def life_stage(species: str, age: int) -> str:
    """Rough life stage used to filter the knowledge base."""
    species = species.lower()
    if age < 1:
        return "puppy"
    if species == "dog" and age >= 8:
        return "senior"
    if species == "cat" and age >= 10:
        return "senior"
    return "adult"


@dataclass
class Advice:
    status: str                      # "grounded" | "fallback" | "urgent"
    message: str
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


class GuidanceAdvisor:
    def __init__(self, retriever: Optional[KnowledgeRetriever] = None,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        self.retriever = retriever or KnowledgeRetriever()
        self.threshold = threshold

    # -- per-task advice ------------------------------------------------------

    def advise_task(self, pet, task) -> Advice:
        """Retrieve guidance for one task and shape it into cited advice."""
        category = task.category.value
        try:
            text = f"{task.description} {category}".lower()
            if any(term in text for term in URGENT_TERMS):
                return self._urgent(pet, task)

            stage = life_stage(pet.species, pet.age)
            # Task words (plus category synonyms) decide WHICH chunk wins. Species and life stage are
            # applied as metadata filters, so only the pet's name and breed notes go in the query.
            task_words = f"{category} {CATEGORY_TERMS.get(category, '')} {task.description}"
            query = f"{task_words} {task_words} {pet.name} {pet.notes}"
            hits = self.retriever.retrieve(query, k=3, species=pet.species,
                                           life_stage=stage, pet_name=pet.name)
            hits = [h for h in hits if h.chunk.tip]
        except Exception:  # guardrail: retrieval must never crash the planner
            logger.exception("advise_task failed for %s / %s", getattr(pet, "name", "?"), task.description)
            return self._with_medication_note(task, Advice("fallback", FALLBACK_MESSAGE))

        if not hits or hits[0].score < self.threshold:
            top = round(hits[0].score, 3) if hits else 0.0
            logger.info("fallback: %s / %s (top score %s below %s)", pet.name, task.description,
                        top, self.threshold)
            return self._with_medication_note(task, Advice("fallback", FALLBACK_MESSAGE, confidence=top))

        best = hits[0]
        tips = [best.chunk.tip]
        citations = [best.chunk.citation]
        for extra in hits[1:2]:
            if (extra.score >= self.threshold and extra.score >= 0.75 * best.score
                    and extra.chunk.tip not in tips):
                tips.append(extra.chunk.tip)
                citations.append(extra.chunk.citation)

        warnings = self._check_limits(task, hits)
        advice = Advice("grounded", " ".join(tips), citations, round(best.score, 2), warnings)
        logger.info("advice: %s / %s -> %s (confidence %.2f, warnings=%d)", pet.name,
                    task.description, citations, best.score, len(warnings))
        return self._with_medication_note(task, advice)

    def _check_limits(self, task, hits: list[Hit]) -> list[str]:
        """If a retrieved chunk states a max duration for this category, compare it to the task."""
        warnings = []
        for hit in hits:
            if hit.score < self.threshold:
                continue
            limit = hit.chunk.max_minutes.get(task.category.value)
            if limit is not None and task.duration_minutes > limit:
                warnings.append(
                    f"{task.duration_minutes} min is longer than the {limit}-min guideline "
                    f"[{hit.chunk.citation}]; consider splitting it into shorter sessions."
                )
                break
        return warnings

    def _with_medication_note(self, task, advice: Advice) -> Advice:
        if task.category.value == "medication" and MEDICATION_REMINDER not in advice.message:
            advice.message = f"{advice.message} {MEDICATION_REMINDER}"
        return advice

    def _urgent(self, pet, task) -> Advice:
        hits = self.retriever.retrieve("emergency signs call vet immediately", k=1)
        tip = hits[0].chunk.tip if hits else "Contact a vet immediately."
        cites = [hits[0].chunk.citation] if hits else []
        logger.warning("URGENT wording in task for %s: %r", pet.name, task.description)
        return Advice("urgent", f"This sounds urgent. {tip}", cites, 1.0)

    # -- missing-task suggestions ---------------------------------------------

    def suggest_tasks(self, pet, existing_tasks) -> list[Suggestion]:
        """Tasks the knowledge base recommends that the pet does not already have.

        Suggestions are proposals only: the owner must approve them (human in the loop).
        """
        try:
            stage = life_stage(pet.species, pet.age)
            query = f"{pet.name} {pet.notes} care"
            hits = self.retriever.retrieve(query, k=None, species=pet.species, life_stage=stage,
                                           pet_name=pet.name, include_zero=True)
        except Exception:
            logger.exception("suggest_tasks failed for %s", getattr(pet, "name", "?"))
            return []

        found: list[Suggestion] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            for category, template, minutes, key in hit.chunk.suggestions:
                description = template.replace("{pet}", pet.name)
                if (category, description.lower()) in seen or self._already_covered(
                        pet, category, description, key, existing_tasks):
                    continue
                seen.add((category, description.lower()))
                found.append(Suggestion(pet.name, category, description, minutes,
                                        hit.chunk.citation, round(hit.score, 2), key))
        logger.info("suggestions for %s: %s", pet.name, [s.description for s in found])
        return found

    @staticmethod
    def _already_covered(pet, category: str, description: str, key: str, existing_tasks) -> bool:
        """True if the pet already has a same-category task that covers this need.

        With a key word (e.g. "teeth") any same-category task mentioning it counts; otherwise
        fall back to more than half of the suggestion's words matching.
        """
        ignore = set(tokenize(pet.name))
        wanted = set(tokenize(description)) - ignore
        if not wanted:
            return False
        for task in existing_tasks:
            if task.category.value != category:
                continue
            have = set(tokenize(task.description)) - ignore
            if key and tokenize(key)[0] in have:
                return True
            if not key and len(wanted & have) / len(wanted) > 0.5:
                return True
        return False
