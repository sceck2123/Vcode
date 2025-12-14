# HexSecGPT (patched)

This folder contains a patched, safer, and more user-friendly version of `HexSecGPT.py` compared to the original upstream script.

> **Disclaimer:** The modifications introduced in this patched version may still contain bugs or edge-case issues. However, across multiple tests the script behaves very well and has proven reliable in normal usage.

Main goals of the patch:
- safer API key handling (avoid plaintext storage),
- correct provider selection (OpenRouter vs DeepSeek),
- immediate key verification before saving/using it,
- “session-only (RAM)” mode,
- improved UX (menu, purge, messages),

---

## What changed vs the original (upstream)

### 1) Provider selection at startup
On startup, the script asks which provider you want to use:
- **OpenRouter**
- **DeepSeek**

This choice affects:
- API base URL,
- model,
- key validation endpoint.

### 2) API key management: Keyring + Session-only (RAM)
Two modes are available:

- **Keyring (persistent)**  
  The key is stored in the OS keyring (on Linux desktop: Secret Service / GNOME Keyring).
  - Service: `HexSecGPT`
  - Account (separated per provider):
    - `HexSecGPT-API:openrouter`
    - `HexSecGPT-API:deepseek`

- **Session-only (RAM)**  
  The key remains only in memory for the current run. It is not written to disk or to the keyring.

**Legacy migration:** if an old `.HexSec` file existed with the key, the script can migrate it to the keyring (and then remove it from `.HexSec`) when keyring is enabled.

### 3) Immediate key verification
When you enter a key, the script performs a lightweight verification:
- **OpenRouter**: uses an authentication-required endpoint, so a DeepSeek key or an invalid key fails immediately.
- **DeepSeek**: verifies through the provider API.

If it fails, you’ll see a clear message like:
> “Invalid key or wrong provider”
and you can re-enter the key before continuing.

### 4) Purge (key deletion) = session expired (Keyring only)
If you delete the key from the keyring via **Purge Stored Key**, the script shows:
> “Session expired”
and exits immediately.

This happens **only** when the key is actually deleted.

### 5) In Session-only (RAM) mode, “Purge Stored Key” is hidden
If you select **Session-only (RAM)**, the main menu hides the “Purge Stored Key (Keyring)” option, because in that mode it must not use the keyring.

---

## Installation

### Python dependencies
Create a venv and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure `requirements.txt` contains at least:
- `openai`
- `rich`
- `python-dotenv`
- `pwinput`
- `colorama`
- `keyring`

### Kali Linux (desktop) – keyring packages (optional)
If you want persistent key storage in the keyring on Kali in a GUI session:
```bash
sudo apt update
sudo apt install -y gnome-keyring seahorse libsecret-1-0
```

To inspect from terminal (optional):
```bash
sudo apt install -y libsecret-tools
```

> If you are on SSH/headless, keyring is often not available: use “session-only (RAM)”.

---

## Run

### Normal run
```bash
python3 HexSecGPT.py
```

### Force “Session-only (RAM)” from the first run
This completely bypasses the keyring:
```bash
HEXSECGPT_SESSION_ONLY=1 python3 HexSecGPT.py
```

In this mode:
- you enter the key once,
- it is verified immediately,
- it remains valid until you close the program,
- it is not stored persistently.

---

## Security notes (realistic threat model)
- The keyring protects against accidental leaks (commits, backups, plaintext files).
- Session-only (RAM) prevents any persistent storage.
- If the machine is already compromised (malware), no software-only solution is perfect: the key can be read from memory or intercepted at input.
