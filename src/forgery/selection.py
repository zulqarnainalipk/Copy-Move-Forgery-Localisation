"""Operating-point selection objectives applied to validation summaries."""

OBJECTIVES = ("mixed", "official", "forged_tnr90", "forged_tnr95")
DESCRIPTIONS = {
    "mixed": "mixed pixel F1 (all images)",
    "official": "competition score",
    "forged_tnr90": "forged-image F1 with authentic true-negative rate of at least 0.90",
    "forged_tnr95": "forged-image F1 with authentic true-negative rate of at least 0.95",
}


def parse(objective):
    """Return (summary column to maximise, minimum authentic true-negative rate or None)."""
    if objective == "mixed":
        return "mixed_pixel_f1", None
    if objective == "official":
        return "official_score", None
    if objective.startswith("forged_tnr"):
        return "forged_pixel_f1", int(objective[len("forged_tnr"):]) / 100
    raise ValueError(f"unknown objective {objective}")


def choose(table, objective, kind="quicksort"):
    """Best row of a table of validation summaries under the objective; rows below the true-negative floor are excluded.

    If no row reaches the floor, the rows with the highest true-negative rate are used. kind is the pandas sort kind;
    "stable" keeps the first of tied rows."""
    column, floor = parse(objective)
    table = table.reset_index(drop=True)
    candidates = table
    if floor is not None:
        feasible = table[table.authentic_tnr >= floor]
        candidates = feasible if len(feasible) else table[table.authentic_tnr == table.authentic_tnr.max()]
    return candidates.sort_values(column, ascending=False, kind=kind).iloc[0]


def suffix(objective, first):
    return "" if objective == first else f"_{objective}"
