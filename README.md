# Enterprise AI Support Platform
## Session 8 of 12 — The Supervisor Orchestrator

A production-grade multi-agent customer support system built with LangGraph, FastAPI, and Google Gemini. Each session in this 12-part series adds a layer of the architecture. This session introduces an LLM-powered supervisor that dynamically decides which worker handles each ticket — replacing the static routing from Session 7 with a hub-and-spoke orchestration model.

---

## Architecture Overview

```
User Ticket
    │
    ▼
┌─────────────┐
│   TRIAGE    │  ingress_node → PII masking, injection detection
│  SUBGRAPH   │  classify_node → technical / billing / fraud / general
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ SUPERVISOR  │  LLM reads full state → SupervisorDecision (structured output)
│    NODE     │  decides: which worker next, or FINISH
└──────┬──────┘
       │
  ┌────┴────────────────┬────────────────┐
  ▼                     ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ TECH SUPPORT │  │ FRAUD HANDLER│  │GENERAL HANDLER│
│  SUBGRAPH    │  │              │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └────────────────►▼◄─────────────────┘
                   SUPERVISOR
                  (every worker returns here)
                       │
                  next_worker == FINISH
                       │
                      END
```

**Key concepts introduced in Session 8:**
- `SupervisorDecision` — Pydantic model enforcing structured LLM output
- `supervisor_node` — reads the complete `SharedState` before every routing decision
- Hub-and-spoke topology — no worker goes directly to END; all return to supervisor
- `delegation_count` safety limit (`MAX_DELEGATIONS = 5`) prevents infinite loops

---

## What Each Session Built

| # | Session | What Was Added |
|---|---------|----------------|
| 1 | The Blueprint | Graph skeleton, state schema, category routing |
| 2 | Tool Binding | CRM lookup, KB search, tool node |
| 3 | The ReAct Architecture | ReAct loop, circuit breaker, duplicate detection |
| 4 | Persistence & Threading | SQLite checkpointer, thread continuity |
| 5 | Context Management | Summarization node, bounded context |
| 6 | Guardrails & Bounding | PII masking (Presidio), injection detection |
| 7 | Multi-Agent Topologies | Triage + tech_support compiled subgraphs |
| **8** | **The Supervisor Orchestrator** | **LLM-powered hub-and-spoke routing** |
| 9 | Shared Scratchpads | Parallel agent fan-out with `Send` |
| 10–12 | Human-in-the-loop, GitHub integration, Audit timeline | — |

---

## Prerequisites

- Python 3.10+
- A Google AI API key (Gemini 2.5 Flash)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/waseemkhan606/phase3-session8
cd phase3-session8
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

Required by Presidio for PII detection:

```bash
python -m spacy download en_core_web_lg
```

### 5. Set your API key

```bash
# Option A — .env file (recommended)
echo "GOOGLE_API_KEY=your-key-here" > .env

# Option B — shell export
export GOOGLE_API_KEY=your-key-here
```

Get a key at: https://aistudio.google.com/app/apikey

---

## Running

### Start the web server

```bash
python api.py
```

Then open **http://localhost:8000** in your browser.

### Run the CLI test suite

```bash
python support_agent.py
```

Runs 5 test cases (billing, technical, multi-intent, safety limit, general) followed by the 5-check Session 8 verification suite. Expected output ends with:

```
SESSION 8 COMPLETE — 5/5 checks passed
  ✅ PASS  Simple ticket — supervisor routes and returns FINISH
  ✅ PASS  delegation_count increments on each supervisor call
  ✅ PASS  supervisor writes routing decisions to internal_notes
  ✅ PASS  Safety limit fires at MAX_DELEGATIONS=1
  ✅ PASS  next_worker field set correctly by supervisor
```

---

## Project Files

```
session8/
├── support_agent.py   # Core agent: graph, nodes, supervisor logic, state
├── api.py             # FastAPI server: /api/run, /api/stream, /api/verify
├── index.html         # Single-file frontend with Supervisor Panel
├── requirements.txt   # Pinned Python dependencies
└── .env               # API key (not committed)
```

**Runtime-generated (not committed):**
```
support.db             # SQLite checkpoint store (conversation threads)
support.db-shm         # SQLite shared memory
support.db-wal         # SQLite write-ahead log
__pycache__/           # Python bytecode cache
.venv/                 # Virtual environment
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/run` | Submit a ticket, get full result |
| `POST` | `/api/stream` | Submit a ticket, stream node-by-node SSE events |
| `POST` | `/api/verify` | Run the Session 8 verification suite |
| `GET`  | `/api/threads` | List all active conversation thread IDs |
| `GET`  | `/api/history/{thread_id}` | Get checkpoint history for a thread |
| `GET`  | `/health` | Server health + config |

### `/api/run` response (Session 8 fields)

```json
{
  "category":         "billing",
  "final_response":   "Your outstanding balance is...",
  "delegation_count": 2,
  "next_worker":      "FINISH",
  "max_delegations":  5,
  "supervisor_notes": [
    {
      "agent":      "supervisor",
      "finding":    "Routed to tech_support",
      "reasoning":  "Billing issue present and not yet addressed",
      "delegation": 1
    }
  ]
}
```

---

## UI Panels

The browser UI at `http://localhost:8000` has these panels (right column, top to bottom):

| Panel | What it shows |
|-------|---------------|
| Execution Inspector | Node-by-node timeline with subgraph badges |
| Active Threads | Thread selector for multi-turn conversations |
| ReAct Loop | Iteration counter and circuit breaker status |
| Conversation History | Checkpoint history for selected thread |
| Security | PII detection, injection check, safety gate |
| Context Summary | Summarization progress and compressed text |
| Tool Calls | Arguments and results for each tool invoked |
| Subgraph Execution | TRIAGE / TECH SUPPORT / SUPERVISOR badges |
| **Supervisor** | Delegation progress bar, last decision, routing log |

### Supervisor Panel

The Supervisor Panel shows:
- **Delegations Used / Max Allowed** — progress bar turns amber at 60 %, red at 100 %
- **Last Decision** — colour-coded: green for FINISH, blue for tech_support, red for fraud_handler, purple for triage
- **Routing log** — one entry per supervisor call with the worker chosen and the LLM's reasoning

---

## Key Constants

| Constant | Default | File | Purpose |
|----------|---------|------|---------|
| `MAX_DELEGATIONS` | `5` | `support_agent.py` | Hard limit on supervisor loops |
| `MAX_ITERATIONS` | `5` | `support_agent.py` | ReAct circuit breaker |
| `SUMMARY_THRESHOLD` | `8` | `support_agent.py` | Messages before summarization fires |
| `CONTEXT_THRESHOLD` | `12` | `support_agent.py` | Context trim threshold |

---

## The Supervisor in Detail

### `SupervisorDecision` model

```python
class SupervisorDecision(BaseModel):
    next_worker: Literal[
        'triage', 'tech_support',
        'fraud_handler', 'general_handler', 'FINISH'
    ]
    reasoning: str
```

Pydantic validates the LLM's output at runtime. If the model returns anything outside the five literals, a `ValidationError` is caught and routing falls back to `triage`.

### Decision rules (in the supervisor prompt)

1. If `final_response` is set and addresses all issues → **FINISH**
2. If billing/technical issue not yet addressed → **tech_support**
3. If suspicious activity not yet investigated → **fraud_handler**
4. If classification seems wrong or unsafe input detected → **triage**
5. If `delegation_count` exceeds 3 → **FINISH**

For rule 3 to work correctly, `fraud_handler` must write a finding to `internal_notes` after it runs. Without this entry the supervisor has no proof the investigation completed and re-routes to `fraud_handler` on every delegation until the loop limit fires. All handlers write to `internal_notes`:

| Handler | `internal_notes` entry |
|---------|------------------------|
| `tech_support_subgraph` | written by `agent_node` via `tool_results` fingerprints |
| `fraud_handler` | `agent: fraud_handler`, risk score + recommendation |
| `general_handler` | calls LLM directly — response is the proof of completion |
| `supervisor_node` | `agent: supervisor`, routing decision + reasoning |

### Safety limit

The safety limit fires **before** the LLM call, consuming zero tokens:

```python
if delegation > MAX_DELEGATIONS:
    return {'delegation_count': delegation, 'next_worker': 'FINISH'}
```

---

## Post-Session Fixes

Two bugs were found and fixed after the initial Session 8 build:

**1. `general_handler` returned a hardcoded stub response**
- Before: always returned `"Thank you for reaching out. Your inquiry has been received…"`
- Problem: the supervisor saw an unanswered ticket on every delegation and kept re-routing to `general_handler` until `MAX_DELEGATIONS` forced a FINISH — every general question got the same useless canned response
- Fix: `general_handler` now calls the LLM with the actual ticket text

**2. `fraud_handler` did not write to `internal_notes`**
- Before: set `final_response` correctly but wrote nothing to `internal_notes`
- Problem: the supervisor's rule 3 ("suspicious activity not yet investigated") kept firing because there was no `fraud_handler` entry in `internal_notes` to prove investigation was complete — fraud tickets looped through 4 delegations before the "exceeds 3" rule forced FINISH
- Fix: `fraud_handler` now appends `{agent: 'fraud_handler', finding: '...', risk_score, recommendation}` to `internal_notes`, so the supervisor sees proof of completion on delegation 2 and returns FINISH immediately

**Lesson:** every worker that returns to the supervisor must write a finding to `internal_notes`. Without it, the supervisor has no memory of what already ran.

---

## Session 9 Preview — Shared Scratchpads & Consensus

Session 9 adds parallel agent execution using LangGraph's `Send` API:

- `dispatcher_node` fans out to tech, billing, and fraud agents simultaneously
- Each agent writes a structured finding to `internal_notes` (the shared scratchpad)
- `synthesizer_node` reads all findings after the superstep and produces one unified response
- The supervisor from Session 8 remains — used for sequential routing when parallel is not needed
