# Live Translation Agent — Overview

## What we're building

A new kind of agent that does **live interpretation**, like a human conference interpreter:

- **One person speaks** (the presenter) into their microphone.
- **Many people listen** by opening a **shareable link**. Each listener picks the language they want and hears the speech translated live into that language.
- Everyone listening in the same language shares one translation stream, so 100 Hindi listeners cost the same to translate as 1.
- The presenter only talks — they don't hear a bot talking back to them.

Plus a small bonus: an **opt-in "public link" switch** you can turn on for *any* agent, so you can share a normal agent with someone without giving them a login.

## How it works, in pictures

```
   Presenter speaks  ─▶  [ transcribe ]  ─▶  translate to Hindi  ─▶  🔊 Hindi listeners
                                          ─▶  translate to Tamil  ─▶  🔊 Tamil listeners
                                          ─▶  translate to English ─▶ 🔊 English listeners
```

- The presenter's voice is turned into text.
- For each language someone is actually listening in, the text is translated and spoken out loud, then sent to everyone on that language.
- A language stream only starts when someone picks it, and stops when the last listener leaves.

## What you'll do as a user

1. **Create the agent.** Pick the new "Live Translation" type. Choose the language the presenter will speak (e.g. English) and tick the languages listeners may choose (e.g. Hindi, Tamil). Turn on "Enable public link".
2. **Get your links.** You get a **listener link** to share with your audience, and a **"Start broadcasting"** button for yourself.
3. **Broadcast.** Click "Start broadcasting" and speak.
4. **Audience listens.** Anyone with the link opens it, picks a language, and hears you live in that language. No login needed.

## Why this design

- **Reuses what exists.** Transcription, translation, and speech are already in the system for normal calls. We reuse all of it. The only genuinely new piece is "send the same audio to many listeners at once."
- **Safe by default.** No agent is public unless you flip the switch. Every public link uses a random token, not the agent's internal ID, so links can't be guessed.
- **Doesn't touch existing calls.** All the new logic is in new files and new links. Normal telephony and browser test calls keep working exactly as before.
- **Simple, not clever.** Translation cost grows with the number of *languages* in use, not the number of listeners — so a big audience is cheap.

## Should we make every agent shareable by default?

**No — make it opt-in.** A public link lets strangers use your agent, which spends your translation/speech budget and could be abused. So we add one on/off switch per agent (off by default). Turn it on when you want to share; leave it off and the agent stays private. This still gives you the "share any agent" ability you asked about — just deliberately, not automatically.

## The one thing to decide before launch

The live "room" that connects a presenter to their listeners lives in one server process. If the voice server runs several processes at once, a presenter and a listener could accidentally land in different ones and not connect.

**Simple first step:** run live translation on a single process (or route by link so everyone in one session lands together). This needs no extra infrastructure. If it ever needs to scale bigger, we can add a shared message layer later — the design already leaves room for that.

## What's a "translation" agent vs the others

| | Conversational | Non-conversational (alert) | **Live Translation (new)** |
|---|---|---|---|
| Talks back to caller | Yes | Plays one message | No — presenter just speaks |
| Who listens | The caller | The caller | Many public listeners |
| Language | One | One | Presenter's + many listener choices |
| Public link | Optional (opt-in) | Optional (opt-in) | Yes, that's the point |

## Rollout in plain steps

1. Add the new agent type and its settings (source language, listener languages, public switch).
2. Add the public link + a page listeners visit to pick a language and listen.
3. Add the presenter's "Start broadcasting" button and mic capture.
4. Add the behind-the-scenes "translate once per language, send to everyone" piece.
5. Test with one presenter and a few listeners on different languages.

Full details, file references, and the task checklist are in `TECHNICAL.md`.
