import sys
from typing import Any

from tqdm.auto import tqdm


class ProgressTracker:
    """Track simple counters and expose them as tqdm postfix metrics."""

    def __init__(self, *, failed: int = 0) -> None:
        self.failed = failed

    def update(self, bar: tqdm, *, n: int = 1, failed: int | None = None, **metrics: Any) -> None:
        if failed is not None:
            self.failed = failed

        postfix = {"failed": self.failed}
        for key, value in metrics.items():
            postfix[key] = _format_metric(value)

        bar.set_postfix(**postfix)
        bar.update(n)


def create_progress_bar(
    *,
    total: int | None = None,
    desc: str = "",
    leave: bool = False,
    unit: str = "it",
    disable: bool | None = None,
    **kwargs: Any,
) -> tqdm:
    """Create a consistent tqdm bar for CLI and notebook use."""
    if disable is None:
        disable = not sys.stdout.isatty()
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        unit=unit,
        disable=disable,
        dynamic_ncols=True,
        **kwargs,
    )


def _format_metric(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value
