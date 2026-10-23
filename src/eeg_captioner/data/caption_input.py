"""Historical optional annotation-input compaction; numeric values round to four significant digits."""
import re

def compact_summary(text: str) -> str:
    """Remove prompt formatting overhead while preserving useful EEG values."""
    compact_lines = []
    skipping_scope = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## **Scope"):
            skipping_scope = True
            continue
        if skipping_scope:
            if line == "---":
                skipping_scope = False
            continue
        if (
            not line
            or line == "---"
            or line == "# **Segment-level EEG Numeric Summary**"
        ):
            continue

        line = line.replace("**", "")
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                continue
            if sum(cell == "NaN" for cell in cells) >= max(1, len(cells) - 2):
                continue
            rounded_cells = []
            for cell in cells:
                if re.fullmatch(
                    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?",
                    cell,
                    re.IGNORECASE,
                ):
                    value = float(cell)
                    cell = f"{value:.4g}"
                rounded_cells.append(cell)
            line = "\t".join(rounded_cells)
        else:
            line = line.lstrip("#").strip()
        compact_lines.append(line)
    return "\n".join(compact_lines)
