---
description: Leverage the Manifold Engine MCP for Proactive Code Analysis and Context Management
---

# Manifold Engine Workflow

When you are asked to analyze, refactor, or understand a repository, you should leverage the Manifold Engine MCP tools to minimize token usage and maximize structural understanding.

## 1. Initial Reconnaissance (Chaos Scanning)
Instead of manually reading through source files or searching randomly, start by running a batch scan to mathematically identify the most fragile parts of the codebase:
```json
{"call": "mcp_manifold-engine_batch_chaos_scan", "arguments": { "pattern": "*.py", "max_files": 10 }}
```
Review the files with the highest `chaos_score` or an `OSCILLATION`/`PERSISTENT_HIGH` collapse risk. These are your primary targets for refactoring.

## 2. Deep Dive (Predictive Ejection)
When investigating a specific file, check its trajectory to see if it requires immediate ejection/rewriting:
```json
{"call": "mcp_manifold-engine_predict_structural_ejection", "arguments": { "path": "path/to/file.py" }}
```

## 3. Persistent Context (Fact Injection)
When the user gives you complex requirements, architecture rules, or context during a task, do not rely purely on your prompt history. Inject them into the Manifold Engine's Working Memory:
```json
{"call": "mcp_manifold-engine_inject_fact", "arguments": { "fact_id": "current_task_architecture", "fact_text": "The project uses XYZ pattern. Requirements: 1, 2, 3..." }}
```
**Important:** Clean up your facts using `mcp_manifold-engine_remove_fact` when the context is no longer relevant to avoid polluting the workspace.

## 4. Live Modifications
If you edit or rewrite files during your task, the native filesystem watcher will automatically re-index them. You can immediately use `mcp_manifold-engine_search_code` to verify the new structure without needing to dump the file to the terminal.
