<!-- zenno:superpowers-redirect -->
## Zenno: Superpowers output location override

When using the superpowers:writing-plans skill, save plans to `zenno/plans/`
instead of the default `docs/superpowers/plans/`.

When using the superpowers:brainstorming skill, save specs to `zenno/specs/`
instead of the default `docs/superpowers/specs/`.
<!-- zenno:superpowers-redirect -->

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues (uses the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default Matt Pocock vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `GLOSSARY.md` and `docs/adr/` at the repo root, plus this skill's consumer rules in `docs/agents/domain.md` (creating glossary/ADRs lazily via `/domain-modeling`).
