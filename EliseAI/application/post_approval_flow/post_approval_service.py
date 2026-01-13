from enum import Enum


class ItemStatus(Enum):
    """Status values for post-approval items."""
    
    Pending = "Pending"
    InProgress = "InProgress"
    Completed = "Completed"
    Failed = "Failed"
