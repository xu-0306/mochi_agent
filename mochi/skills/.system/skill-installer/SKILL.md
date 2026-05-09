---
name: skill-installer
description: Install or add Mochi skills by placing Codex/Claude-style skill directories under the configured skills directory.
tags: [skills, install, import, SKILL.md]
---

# Skill Installer

Use this skill when the user asks to install, add, import, or enable a skill.

## Mochi Skill Layout

Mochi loads filesystem skills from the configured `skills_dir`. A filesystem skill is a directory that contains a `SKILL.md` file:

```text
~/.mochi/skills/
  example-skill/
    SKILL.md
    scripts/
      helper.py
```

`SKILL.md` should use YAML frontmatter followed by Markdown instructions:

```markdown
---
name: example-skill
description: One sentence describing when this skill should be used.
tags: [example, workflow]
---

# Example Skill

Instructions for the agent.
```

## Behavior

- No manual database import is required.
- On skill list/search/chat, Mochi scans `skills_dir/**/SKILL.md`.
- The SQLite `skills.db` file is only a searchable cache and can be rebuilt from the files.
- To add a skill manually, copy or create the skill directory under `skills_dir`.
- To update a skill, edit its `SKILL.md`; Mochi detects content changes and updates the index.
- To remove a filesystem skill, delete its skill directory; Mochi removes it from the index on the next sync.

## Notes

- Keep helper scripts and assets inside the same skill directory.
- Prefer concise trigger-focused `description` and `tags`; they are used for search.
- Do not put secrets in `SKILL.md`.
