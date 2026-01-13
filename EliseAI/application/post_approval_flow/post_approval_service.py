from enum import Enum


class ItemStatus(str, Enum):
    """Status values for post-approval items."""

    PENDING = "Pending"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"
