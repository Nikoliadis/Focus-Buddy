<div align="center">

# FocusBuddy

**Real-time study rooms with a synchronized Pomodoro timer, live chat and an AI study assistant.**

[![CI](https://github.com/Nikoliadis/Focus-Buddy/actions/workflows/ci.yml/badge.svg)](https://github.com/Nikoliadis/Focus-Buddy/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/flask-3.0-000000?logo=flask)
![PostgreSQL](https://img.shields.io/badge/postgresql-16-4169E1?logo=postgresql&logoColor=white)
![Code style: ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)

[**Live demo → thefocusbuddy.org**](https://thefocusbuddy.org)

<img src="docs/screenshots/demo.gif" alt="Two browsers in the same room: the owner starts the timer and both see it count down in sync, and chat messages appear instantly on both sides" width="100%">

<sub>Two users in the same room: the timer and chat stay in sync over WebSockets.</sub>

</div>

---

## Features

| | |
|---|---|
| **Study rooms** | Public or password-protected rooms with shareable 8-character join codes |
| **Synchronized Pomodoro** | Server-authoritative timer with focus/break cycles; everyone sees the same countdown |
| **Live presence** | Online / last-seen indicators for every member, pushed over Socket.IO |
| **Room chat** | Persistent chat history, owner moderation (mute / unmute / kick) |
| **FocusBot AI** | Study assistant powered by Llama 3.1 through the Groq API, with conversation history |
| **Voice & screen sharing** | Peer-to-peer WebRTC; the server only relays signaling, no media goes through it |
| **Stats & leaderboard** | Daily/weekly totals, streaks, daily goals, per-room breakdown, CSV export |
| **Accounts** | Email verification and password reset (Resend), rate-limited auth endpoints |
| **Tasks** | Personal to-do list with public/private items, shown next to the room |

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/room.png" alt="Study room with tasks, members, timer and chat"><p align="center"><b>Study room</b></p></td>
    <td width="50%"><img src="docs/screenshots/stats.png" alt="Personal statistics dashboard"><p align="center"><b>Statistics</b></p></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/screenshots/rooms.png" alt="Room list with public rooms"><p align="center"><b>Rooms</b></p></td>
    <td width="50%"><img src="docs/screenshots/leaderboard.png" alt="Global leaderboard"><p align="center"><b>Leaderboard</b></p></td>
  </tr>
</table>

## Architecture

```mermaid
flowchart LR
    subgraph Browser
        UI["Jinja templates<br/>+ vanilla JS"]
        SIO["Socket.IO client"]
        RTC["WebRTC peer"]
    end

    subgraph Server["Flask app (gunicorn + gevent)"]
        direction TB
        HTTP["HTTP blueprints<br/>auth · rooms · main"]
        WS["Socket.IO handlers<br/>presence · timer · chat · signaling"]
        SVC["Service layer<br/>rooms · sessions · stats"]
        MEM[("In-memory presence<br/>& cycle state")]
        HTTP --> SVC
        WS --> SVC
        WS --> MEM
    end

    DB[("PostgreSQL<br/>SQLAlchemy ORM")]
    GROQ["Groq API<br/>Llama 3.1 8B"]
    RESEND["Resend<br/>email API"]

    UI -->|"forms / fetch (CSRF-protected)"| HTTP
    SIO <-->|"WebSocket events"| WS
    RTC <-.->|"media, peer-to-peer"| RTC2["Other members"]
    WS -.->|"SDP / ICE relay"| SIO
    SVC --> DB
    HTTP -->|"FocusBot chat"| GROQ
    HTTP -->|"verify / reset emails"| RESEND
```

### How the synchronized timer works

The server is the single source of truth: it stores `started_at` and accumulated `paused_seconds`, never a ticking counter. Each client receives the remaining time and counts down locally, so late joiners and reconnects always land on the correct second.

```mermaid
sequenceDiagram
    participant O as Owner
    participant S as Server
    participant DB as PostgreSQL
    participant M as Members

    O->>S: timer:start {minutes: 25}
    S->>S: check owner permission
    S->>DB: INSERT focus_session (started_at = now)
    S-->>O: timer:update {status: running, remaining: 1500}
    S-->>M: timer:update {status: running, remaining: 1500}
    Note over O,M: Every client counts down locally from the server value
    M->>S: room:join (late joiner)
    S->>DB: remaining = duration − (now − started_at − paused_seconds)
    S-->>M: timer:update {remaining: 1342}
    O->>S: timer:cycle_next
    S->>DB: end session → write focus_log for stats
    S-->>O: timer:update {phase: break}
    S-->>M: timer:update {phase: break}
```

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, Flask 3, Flask-SocketIO, gevent |
| Database | PostgreSQL 16, SQLAlchemy 2 (psycopg 3) |
| Real-time | WebSockets (Socket.IO), WebRTC with STUN |
| AI | Groq API, Llama 3.1 8B Instant |
| Security | CSRF tokens, rate limiting (Flask-Limiter), hashed passwords, open-redirect protection |
| Email | Resend API |
| Testing | pytest, pytest-cov, Socket.IO test client |
| Tooling | ruff, GitHub Actions, Docker |
| Deployment | Railway (Docker, gunicorn) |

## Run locally

### Option 1: Docker (recommended)

You only need Docker installed.

```bash
git clone https://github.com/Nikoliadis/Focus-Buddy.git
cd Focus-Buddy
docker compose up --build
```

Open <http://localhost:5000>. In a second terminal, load demo data (users, rooms, chat and two weeks of focus history):

```bash
docker compose exec web python -m scripts.seed_demo
```

Log in as **`demo` / `demo1234`**. To try the real-time features, open a private window and log in as `maria` / `demo1234` in the same room.

> Port already in use? Run `DB_PORT=5433 WEB_PORT=5001 docker compose up --build`.

### Option 2: Plain Python with SQLite (no database server needed)

```bash
git clone https://github.com/Nikoliadis/Focus-Buddy.git
cd Focus-Buddy
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

In `.env`, switch `DATABASE_URL` to the SQLite line (`sqlite:///focusbuddy.db`), then:

```bash
python -m scripts.seed_demo
python app.py
```

Open <http://localhost:5000> and log in as **`demo` / `demo1234`**.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | yes | Flask session signing key |
| `DATABASE_URL` | yes | PostgreSQL (`postgresql://…`, Railway format accepted) or SQLite URL |
| `COOKIE_SECURE` | no | `1` in production behind HTTPS |
| `GROQ_API_KEY` | no | Enables FocusBot; without it the assistant returns *"AI not configured"* |
| `RESEND_API_KEY` | no | Enables verification and reset emails. Without it, newly registered users can't verify, so use the seeded demo accounts locally |

## Testing

The test suite covers 130+ cases across authentication, CSRF, room permissions, the timer state machine, statistics and streaks, the REST API, the FocusBot endpoint (with the Groq client mocked), and the Socket.IO events (presence, timer broadcast, chat, moderation, WebRTC signaling).

```bash
pip install -r requirements-dev.txt
pytest --cov            # runs against in-memory SQLite by default
ruff check .
```

To run the suite against PostgreSQL, as CI does:

```bash
TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/focusbuddy_test pytest
```

Every push and pull request runs three GitHub Actions jobs: **ruff lint**, **pytest on a PostgreSQL 16 service container** with coverage, and a **Docker image build**.

## Project structure

```
.
├── app.py                 # App factory, CSRF hook, error handlers
├── config.py              # Env-driven configuration
├── auth/                  # Register, login, email verification, password reset, profile
├── rooms/
│   ├── routes.py          # Room CRUD, membership, session control (HTTP)
│   ├── sockets.py         # Socket.IO: presence, timer, chat, mute, WebRTC signaling
│   ├── service.py         # Room business logic
│   └── sessions_service.py# Pomodoro session state machine
├── main/
│   ├── routes.py          # Home, stats, leaderboard, todos API, FocusBot API
│   ├── stats_service.py   # Aggregations, streaks, leaderboard
│   └── csrf.py · limiter_ext.py · email_utils.py · …
├── models/                # SQLAlchemy models
├── templates/ · static/   # Jinja templates, CSS, JS
├── scripts/seed_demo.py   # Demo data generator
├── tests/                 # pytest suite
└── .github/workflows/     # CI pipeline
```

## Author

**[@Nikoliadis](https://github.com/Nikoliadis)** · built as my undergraduate thesis project.
[Live app](https://thefocusbuddy.org)
