"""Tests for the RAG layer: retrieval, guardrails, and Scheduler integration.

Run from the project root:  python -m pytest -v
"""

import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advisor import FALLBACK_MESSAGE, GuidanceAdvisor, life_stage  # noqa: E402
from pawpal_system import (Category, Frequency, Owner, Pet, Priority, Scheduler,  # noqa: E402
                           Task)
from retriever import KnowledgeRetriever, configure_logging, load_chunks  # noqa: E402

DAY = date(2026, 11, 17)


@pytest.fixture(scope="module")
def retriever():
    return KnowledgeRetriever()


@pytest.fixture()
def advisor(retriever):
    return GuidanceAdvisor(retriever)


def make_task(description, category, minutes=10, hour=9, must_do=False):
    return Task(description, datetime(2026, 11, 17, hour), minutes, category,
                Priority.MEDIUM, Frequency.DAILY, must_do=must_do)


def chester():
    return Pet("Chester", "dog", 12, notes="yorkie schnauzer mix")


# -- knowledge base loading ---------------------------------------------------

def test_every_chunk_has_a_tip_and_a_source():
    chunks = load_chunks(Path("knowledge"))
    assert len(chunks) >= 30
    assert all(c.tip and c.source.endswith(".md") for c in chunks)


def test_structured_lines_are_parsed_not_indexed_as_prose():
    walks = [c for c in load_chunks(Path("knowledge")) if c.heading == "Chester's walks"][0]
    assert walks.max_minutes == {"walk": 20}
    assert "Max-minutes" not in walks.text


# -- retrieval correctness ----------------------------------------------------

def test_senior_dog_walk_query_finds_walk_guidance(retriever):
    hits = retriever.retrieve("walk exercise senior", k=3, species="dog", life_stage="senior")
    assert hits and any("walk" in h.chunk.heading.lower() for h in hits)


def test_species_filter_keeps_dog_advice_away_from_cats(retriever):
    hits = retriever.retrieve("walk exercise teeth feeding", k=None, species="cat")
    assert hits
    assert all(h.chunk.species in ("any", "cat") or "cat" in h.chunk.species for h in hits)


def test_life_stage_filter_excludes_puppy_advice_for_senior_dogs(retriever):
    hits = retriever.retrieve("puppy vaccine", k=None, species="dog", life_stage="senior")
    assert all(h.chunk.life_stage != "puppy" for h in hits)


def test_other_pets_never_get_chesters_profile(retriever):
    hits = retriever.retrieve("chester walk", k=None, species="dog", pet_name="Bella")
    assert all(h.chunk.pet == "" for h in hits)


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("   ") == []


def test_scores_are_between_zero_and_one_and_sorted(retriever):
    hits = retriever.retrieve("senior dog dental teeth", k=5, species="dog")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1 for s in scores)


# -- advisor: grounded answers and guardrails ---------------------------------

def test_walk_for_chester_cites_his_profile_and_warns_when_too_long(advisor):
    advice = advisor.advise_task(chester(), make_task("Long walk", Category.WALK, 40))
    assert advice.status == "grounded"
    assert any("chester_profile.md" in c for c in advice.citations)
    assert advice.warnings and "20-min guideline" in advice.warnings[0]


def test_short_walk_has_no_warning(advisor):
    advice = advisor.advise_task(chester(), make_task("Short walk", Category.WALK, 15))
    assert advice.status == "grounded"
    assert advice.warnings == []


def test_unknown_species_falls_back_instead_of_guessing(advisor):
    hamster = Pet("Nibbles", "hamster", 1)
    advice = advisor.advise_task(hamster, make_task("Refresh bedding", Category.GROOMING))
    assert advice.status == "fallback"
    assert FALLBACK_MESSAGE in advice.message
    assert advice.citations == []


def test_urgent_wording_tells_owner_to_call_a_vet(advisor):
    advice = advisor.advise_task(chester(), make_task("Chester had a seizure", Category.APPOINTMENT))
    assert advice.status == "urgent"
    assert "urgent" in advice.message.lower()


def test_medication_always_carries_the_vet_reminder_even_on_fallback(advisor):
    hamster = Pet("Nibbles", "hamster", 1)
    for pet in (chester(), hamster):
        advice = advisor.advise_task(pet, make_task("Morning pill", Category.MEDICATION, 5))
        assert "never recommends doses" in advice.message


def test_retrieval_failure_degrades_to_fallback_not_a_crash(advisor, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("index corrupted")
    monkeypatch.setattr(advisor.retriever, "retrieve", boom)
    advice = advisor.advise_task(chester(), make_task("Walk", Category.WALK))
    assert advice.status == "fallback"


def test_missing_knowledge_folder_does_not_crash(tmp_path):
    advisor = GuidanceAdvisor(KnowledgeRetriever(tmp_path / "does_not_exist"))
    advice = advisor.advise_task(chester(), make_task("Walk", Category.WALK))
    assert advice.status == "fallback"
    assert advisor.suggest_tasks(chester(), []) == []


def test_life_stage_boundaries():
    assert life_stage("dog", 0) == "puppy"
    assert life_stage("dog", 5) == "adult"
    assert life_stage("dog", 12) == "senior"
    assert life_stage("cat", 12) == "senior"


# -- suggestions (human in the loop) ------------------------------------------

def test_suggests_teeth_brushing_when_missing_and_skips_it_when_present(advisor):
    pet = chester()
    assert any("teeth" in s.description.lower() for s in advisor.suggest_tasks(pet, []))
    have = [make_task("Brush Chester's teeth", Category.GROOMING, 5)]
    assert not any("teeth" in s.description.lower() for s in advisor.suggest_tasks(pet, have))


def test_other_coat_brushing_does_not_hide_the_teeth_suggestion(advisor):
    have = [make_task("Brush coat", Category.GROOMING, 10)]
    assert any("teeth" in s.description.lower() for s in advisor.suggest_tasks(chester(), have))


def test_hamster_gets_no_cat_or_dog_suggestions(advisor):
    assert advisor.suggest_tasks(Pet("Nibbles", "hamster", 1), []) == []


# -- integration with the Scheduler -------------------------------------------

def build_scheduler(advisor):
    owner = Owner("Andrea", available_minutes=120)
    pet = chester()
    owner.add_pet(pet)
    pet.add_task(make_task("Long walk", Category.WALK, 40))
    return Scheduler(owner, advisor), pet


def test_plan_without_advisor_is_unchanged():
    scheduler, _ = build_scheduler(None)
    item = scheduler.build_daily_plan(DAY).items[0]
    assert item.citations == [] and "Guidance" not in item.reason


def test_plan_with_advisor_changes_reason_and_adds_citations(advisor):
    scheduler, _ = build_scheduler(advisor)
    plan = scheduler.build_daily_plan(DAY)
    item = plan.items[0]
    assert "Guidance:" in item.reason
    assert item.citations and item.warnings
    assert "sources:" in plan.summary() and "warning:" in plan.summary()


def test_accepting_a_suggestion_adds_a_conflict_free_task(advisor):
    scheduler, pet = build_scheduler(advisor)
    suggestion = next(s for s in scheduler.build_daily_plan(DAY).suggestions if "teeth" in s.description)
    before = len(pet.tasks)
    task = scheduler.accept_suggestion(suggestion, DAY)
    assert len(pet.tasks) == before + 1
    assert task.category == Category.GROOMING
    assert scheduler.detect_conflicts() == []


# -- logging ------------------------------------------------------------------

def test_retrievals_are_logged_to_file(tmp_path, retriever):
    log_file = tmp_path / "rag.log"
    logger = logging.getLogger("pawpal.rag")
    old_handlers = logger.handlers[:]
    logger.handlers.clear()
    try:
        configure_logging(str(log_file))
        retriever.retrieve("senior dog walk", k=1, species="dog")
        for handler in logger.handlers:
            handler.flush()
        assert "retrieve:" in log_file.read_text()
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = old_handlers


# -- plan table helpers (used by the Streamlit Daily plan tab) ---------------

def test_plan_rows_show_sources_and_confidence_only_when_rag_is_on(advisor):
    from plan_view import plan_rows, plan_warnings
    scheduler, _ = build_scheduler(advisor)
    plan = scheduler.build_daily_plan(DAY)
    rag_row = plan_rows(plan, show_reasons=True, show_sources=True)[0]
    plain_row = plan_rows(plan, show_reasons=False, show_sources=False)[0]
    assert "chester_profile.md" in rag_row["Sources"] and rag_row["Confidence"] != "n/a"
    assert "Why" in rag_row and "Why" not in plain_row and "Sources" not in plain_row
    assert plan_warnings(plan) and plan_warnings(plan)[0].startswith("Chester - Long walk")


def test_plan_rows_say_so_when_nothing_reliable_was_found(advisor):
    from plan_view import plan_rows
    owner = Owner("Andrea")
    hamster = Pet("Nibbles", "hamster", 1)
    owner.add_pet(hamster)
    hamster.add_task(make_task("Refresh bedding", Category.GROOMING))
    row = plan_rows(Scheduler(owner, advisor).build_daily_plan(DAY), show_sources=True)[0]
    assert row["Sources"].startswith("none") and row["Confidence"] == "n/a"
