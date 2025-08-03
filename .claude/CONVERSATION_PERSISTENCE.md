# Conversation Persistence System

## Overview

This system ensures conversation continuity between Claude sessions by automatically saving and loading conversation history.

## Methods for Persistence

### 1. Session Logs (Implemented)
Location: `.claude/sessions/`
- Each conversation saved as timestamped markdown
- Includes summary, decisions, code changes, TODOs
- Human-readable and git-trackable

### 2. Hooks System (Claude Settings)
You can configure hooks in Claude's settings to automatically:
- Save conversation at regular intervals
- Export on session end
- Load context on session start

Example `.claude/hooks.json`:
```json
{
  "onSessionStart": {
    "loadContext": ".claude/sessions/current_session.md"
  },
  "onSessionEnd": {
    "saveSession": ".claude/sessions/session_${timestamp}.md"
  }
}
```

### 3. CLAUDE.md Context Loading
Add to CLAUDE.md:
```markdown
## Session Context
When starting a conversation, check:
1. `.claude/sessions/` for recent session logs
2. Load the latest session summary
3. Continue from previous TODOs
```

### 4. Automated Summary Generation
At the end of each conversation, create:
1. Detailed session log
2. Update session_summary.md
3. Update current_context.md

## Best Practices

### For Users:
1. **Before closing session**: Ask Claude to "save session summary"
2. **When reopening**: Say "load previous session" or "continue from last session"
3. **Regular saves**: Periodically ask for session updates

### For Claude:
1. **Check for context**: Always look for `.claude/sessions/` on start
2. **Summarize regularly**: Create summaries of major accomplishments
3. **Track decisions**: Log all important decisions and rationale
4. **Update TODOs**: Maintain running list of next steps

## Session File Format

```markdown
# Session Log: [Date]

## Metadata
- Date: 
- Time:
- Branch:
- Focus:

## Previous Context
[Summary from last session]

## This Session
### Topics Discussed
1. [Topic 1]
2. [Topic 2]

### Code Changes
- Created: [files]
- Modified: [files]
- Deleted: [files]

### Decisions Made
1. [Decision + rationale]
2. [Decision + rationale]

### Commands Run
```bash
[Important commands]
```

### Next Steps
- [ ] TODO 1
- [ ] TODO 2

## Context for Next Session
[What Claude needs to know]
```

## Implementation Status

✅ Created `.claude/sessions/` directory
✅ Created first session log
✅ Created this persistence guide
⏳ Hook system requires Claude settings configuration
⏳ Automatic loading requires user prompt

## Usage Instructions

### Saving a Session:
```
User: "Please save our conversation summary"
Claude: [Creates detailed session log with all context]
```

### Loading Previous Session:
```
User: "Please load our previous session context"
Claude: [Reads latest session log and continues]
```

### Best Pattern:
1. Start: "Load previous session from .claude/sessions/"
2. Work: Normal conversation
3. End: "Save session summary before I close"

This ensures perfect continuity between conversations!