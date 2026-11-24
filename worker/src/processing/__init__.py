"""The processor framework: how a processing stage plans, computes and stores a batch."""

from .base import Item, ItemResult, Processor
from .context import ProcessingContext, Progress
from .runner import StageSummary, run_stage

__all__ = [
    "Item",
    "ItemResult",
    "ProcessingContext",
    "Processor",
    "Progress",
    "StageSummary",
    "run_stage",
]
