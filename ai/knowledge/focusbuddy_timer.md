# Synchronized Pomodoro timer

## How the shared timer works
Every room has one shared focus timer. Everyone in the room sees the same countdown at the same time, and people who join late or reconnect see the correct remaining time, because the server keeps the timer and each browser counts down from it.

## Who controls the timer
Only the room owner can start, pause, resume, reset or end a focus session. Other members follow the owner's timer.

## Focus and break lengths
When starting a session the owner chooses the focus length and the break length. Focus can be between 1 and 180 minutes (default 25) and a break between 1 and 60 minutes (default 5).

## Focus and break cycles
When a focus block finishes, everyone hears a sound and gets a browser notification. The owner then sees a five-second countdown before the break starts automatically, with a Cancel button to stop it. When the break ends, the next focus block starts the same way and the cycle number goes up by one.

## Pausing, resetting and ending
Pause freezes the countdown for everyone and Resume continues it; paused time does not count as focus. Reset puts the timer back to the start without recording the session. End stops the session and records its focused time; at the moment the session is credited to the room owner's statistics.
