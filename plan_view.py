"""Presentation helpers for the Daily plan tab (plain Python, so they can be unit-tested without Streamlit)."""

from __future__ import annotations


def plan_rows(plan, show_reasons: bool = True, show_sources: bool = False) -> list[dict]:
    """Rows for the plan table. Sources/confidence columns only appear when RAG is on."""
    rows = []
    for item in plan.items:
        row = {
            "Time": f"{item.task.due_time:%H:%M}-{item.task.end_time():%H:%M}",
            "Pet": item.task.pet_name,
            "Task": item.task.description,
            "Priority": item.task.priority.name.title(),
        }
        if show_reasons:
            row["Why"] = item.reason
        if show_sources:
            row["Sources"] = "; ".join(item.citations) if item.citations else "none (no reliable match)"
            row["Confidence"] = f"{item.confidence:.2f}" if item.confidence is not None else "n/a"
        rows.append(row)
    return rows


def plan_warnings(plan) -> list[str]:
    """One line per guideline warning, prefixed with the pet and task it belongs to."""
    return [f"{item.task.pet_name} - {item.task.description}: {warning}"
            for item in plan.items for warning in item.warnings]
