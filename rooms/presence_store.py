from datetime import datetime

# room_id -> set of online user_ids
# In-memory only; resets on server restart (acceptable for this use case)
PRESENCE: dict[int, set[int]] = {}

# user_id -> last seen datetime (set on disconnect or room:leave)
LAST_SEEN: dict[int, datetime] = {}

# room_id -> {"phase": "focus"|"break", "count": int}
ROOM_CYCLES: dict[int, dict] = {}
