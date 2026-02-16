# Session Logs

This directory contains conversation logs with Claude to maintain context between sessions.

## Structure

Each session is saved as a markdown file with timestamp:
- `session_YYYY-MM-DD_HH-MM.md` - Individual session logs
- `current_session.md` - Symlink to the latest session
- `session_summary.md` - High-level summary of all sessions

## Usage

When starting a new conversation:
1. Claude should check for the latest session log
2. Load context from previous conversations
3. Continue where we left off

## Format

Each session log contains:
- Session metadata (date, time, branch)
- Conversation summary
- Key decisions made
- Code changes implemented
- TODOs for next session