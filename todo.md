# TODO

## inference

1. Send the context together with the image.
2. Support working with links.
3. Split triggers into separate skills.
4. Simplify the prompt by splitting the main prompt into skills, with a Ollamadetailed description of what belongs in `signal_category`.
5. Add specific lists of signals.
6. Come up with a solution for RAG (mandatory).
7. Run the neural network in two passes: the first describes the image, the second analyzes it. Evaluate and estimate the computational cost.

## vision_app

### Optional

- Add the ability to upload a profile picture.

## chat

### Tools

1. `analysis_stats` — aggregation with `group_by` (`risk_level | category | day | user`) on top of `count_only`. Read-only, numbers only.
2. `queue_status` — queue length, running and failed jobs with reasons. Users see their own jobs, admins see all.
3. `analyze_link` — **on hold, may be removed**: link analysis is not well tested and there are no suitable reference images. Do not build a chat tool for it until this is decided.
4. `system_health` (staff only) — available backends, default backend and model.
5. Categories (the `Category` table is sent to the analyzer as an overlay on every `/analyze`, so these texts are instructions for the vision model for all users):
   - `list_categories` / `get_category` — read-only: name, title, summary, `is_active`, position; full rules by name. Lets the model explain why an image was flagged.
   - `manage_category` — create / update / toggle / reorder / delete through a confirmation card (reuse `ChatAction`). Permission: staff, same as `staff_required` in the panel, so it does not add to the head-admin dependency.
     - New categories are created with `is_active = False` (draft); activation is a separate card.
     - The card for text fields shows "before → after"; for a new category it shows the full text.
     - Validate with the same rules as `CategoryForm` (name: latin/digits/`_`, unique).
     - Keep the previous state in the `ChatAction` payload, so a change can be rolled back.
6. Actions with a confirmation card (reuse `ChatAction`):
   - delete / rename / requeue an analysis, cancel a queue job;
   - change the default backend/model and runtime settings (head admin, show "before → after").
7. `web_search` — **later, only as a helper for drafting categories**, not as a general tool:
   - Search results are untrusted text, and a category text ends up in the vision prompt for everyone (persistent prompt injection). Add `web_search` to `_DATA_TOOLS`, so `manage_category` is closed in the same turn: the model writes a draft in the reply, and a human asks to create it in the next message.
   - Created drafts stay inactive until a human activates them.
   - Prefer the provider-native search (Gemini grounding) over our own fetcher: no SSRF surface. Check that the `interactions` API supports it and what the Free Tier quota is.
   - Consider a domain allowlist (reference databases, encyclopedias, official sites).

### Rules for new tools

- Every write goes through a confirmation card; no tool changes state directly.
- If a tool result contains user-generated text (descriptions, nicknames), add the tool to `_DATA_TOOLS` in `runner.py`, so write tools are closed for the rest of that turn.
- Every tool call costs one extra Gemini request (Free Tier: 5 per minute). Prefer a few broad tools over many narrow ones.
- Not planned: arbitrary URL fetch (SSRF), calculator.

### Reliability

8. Store a short marker for turns with cards (e.g. `[action #N created]`) in the chat history, so the model stops copying "I prepared the request" from history without calling the tool. The phantom-card guard in `runner.py` is only a stopgap.
9. Show a clear UI message for 429/503 from the model instead of a bare 502 (include "retry in N s").
