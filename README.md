# FocusBuddy

A real-time collaborative study app built for students.  
Live at: [thefocusbuddy.org](https://thefocusbuddy.org)

## Features

- **Real-time study rooms** — create or join rooms with a unique join code
- **Synchronized Pomodoro timer** — everyone in the room sees the same timer
- **Live presence** — see who's online and who's currently focusing
- **Room chat** — real-time messaging with chat mute (owner controls)
- **My Tasks** — personal to-do list (public/private) inside each room
- **FocusBot AI** — built-in study assistant powered by Llama 3.1 via Groq
- **Screen sharing & microphone** — WebRTC peer-to-peer communication
- **Statistics & leaderboard** — track focus time and compare with others
- **Email verification & password reset** — via Resend
- **Light/Dark mode**

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-SocketIO |
| Database | PostgreSQL (psycopg3) |
| Real-time | WebSockets (gevent) |
| Peer-to-peer | WebRTC (STUN) |
| AI | Groq API (Llama 3.1) |
| Email | Resend API |
| Deployment | Railway |
| Frontend | Vanilla JS, CSS |

## Run Locally

```bash
git clone https://github.com/Nikoliadis/Focus-Buddy.git
cd Focus-Buddy
pip install -r requirements.txt
