function fmt(seconds) {
  const m = String(Math.floor(seconds / 60)).padStart(2, "0");
  const s = String(Math.floor(seconds % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

async function copyInviteLink() {
  const input = document.getElementById("inviteLink");
  if (!input) return;

  try {
    input.select();
    input.setSelectionRange(0, 99999);

    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(input.value);
    } else {
      document.execCommand("copy");
    }

    input.blur();
  } catch (_) {
  }
}

async function pollSession(sessionUrl) {
  try {
    const res = await fetch(sessionUrl, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();

    const statusEl = document.getElementById("fs-status");
    const timerEl = document.getElementById("fs-timer");

    if (statusEl) statusEl.textContent = data.status || "idle";
    if (timerEl) timerEl.textContent = fmt(data.remaining_seconds ?? 1500);
  } catch (_) {
  }
}

async function pollPresence(presenceUrl) {
  try {
    const res = await fetch(presenceUrl, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();

    const countEl = document.getElementById("members-count");
    if (countEl) countEl.textContent = data.count ?? 0;
  } catch (_) {
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const flashes = document.querySelectorAll(".flash");
  flashes.forEach((f) => {
    const close = () => {
      f.classList.add("is-hiding");
      f.addEventListener("animationend", () => f.remove(), { once: true });
    };

    f.addEventListener("click", close);
    f.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === "Escape") close();
    });
  });

  if (flashes.length) {
    setTimeout(() => {
      flashes.forEach((f) => {
        f.classList.add("is-hiding");
        f.addEventListener("animationend", () => f.remove(), { once: true });
      });
    }, 3500);
  }

  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type='submit']");
      if (!btn) return;
      btn.disabled = true;
      btn.dataset.originalText = btn.textContent || "";
      btn.textContent = "Loading…";
    });
  });

  window.copyInviteLink = copyInviteLink;

  const roomPage = document.getElementById("room-page");
  if (roomPage) {
    const sessionUrl = roomPage.dataset.sessionUrl;
    const presenceUrl = roomPage.dataset.presenceUrl;

    if (sessionUrl) {
      pollSession(sessionUrl);
      setInterval(() => pollSession(sessionUrl), 1000);
    }
    if (presenceUrl) {
      pollPresence(presenceUrl);
      setInterval(() => pollPresence(presenceUrl), 5000);
    }
  }
});
