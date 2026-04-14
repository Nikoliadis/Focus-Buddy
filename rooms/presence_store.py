from datetime import datetime
from typing import Dict, Set

# room_id -> set of online user_ids
# In-memory only; resets on server restart (acceptable for this use case)
PRESENCE: Dict[int, Set[int]] = {}

# user_id -> last seen datetime (set on disconnect or room:leave)
LAST_SEEN: Dict[int, datetime] = {}
