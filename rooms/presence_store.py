from typing import Dict, Set

# room_id -> set of online user_ids
# In-memory only; resets on server restart (acceptable for this use case)
PRESENCE: Dict[int, Set[int]] = {}
