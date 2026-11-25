# AI Assistant Guidelines — Ponytail Architecture Rules

> **Core Philosophy**: *"Think like the laziest senior dev in the room. The best code is the code you never wrote."*

Every contributor and AI coding assistant working on **AccessiCaption** must strictly adhere to the **Ladder of Simplicity** before adding or modifying code.

---

## The Ladder of Simplicity

Before writing any new function, class, abstraction, or dependency, evaluate:

1. **Does this need to exist at all?** (YAGNI — You Aren't Gonna Need It)
   - If a feature isn't directly needed for v1 WCAG alt-text generation, skip it.
   - Avoid speculative architectures, abstract factory patterns, and premature generalizations.
2. **Can the standard library do it?**
   - Prefer Python's `math`, `re`, `urllib`, `collections`, `dataclasses`, and `pathlib` over third-party packages.
3. **Does an already-installed dependency solve it?**
   - Check if `torch` or `transformers` already provides the functionality before adding a new package to `requirements.txt`.
4. **Can this be written in a single function or one-liner?**
   - Avoid multi-file class hierarchies when a single pure function suffices.
5. **Only then:** Write the absolute minimum code that works.

---

## "Lazy, Not Negligent" Constraints

Minimalism does **not** mean cutting critical safety or domain value:

- **Accessibility is non-negotiable:** Never weaken or bypass WCAG 2.1/2.2 alt-text normalization (`data/wcag_normalize.py`).
- **Fail gracefully:** Model calls must handle errors cleanly instead of crashing.
- **Annotate necessary shortcuts:** If using a lightweight standard-library heuristic in place of a heavy ML model, label it with `# ponytail: [reason]`.
