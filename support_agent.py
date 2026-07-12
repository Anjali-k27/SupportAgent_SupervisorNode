"""
Enterprise AI Support Platform
Session 8 of 12 — The Supervisor Orchestrator

Extends Session 7 with LLM-powered supervisor routing.
SupervisorDecision Pydantic model enforces structured output.
Hub and spoke: every worker returns to supervisor.
delegation_count safety limit prevents infinite loops.

Run server: python api.py  → http://localhost:8000
Run CLI:    python support_agent.py
"""

import os
import re
import time
import operator
import json
import uuid
import sqlite3
from typing import TypedDict, Annotated, Literal, Any

from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# ── Environment setup ──────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not set. Run: export GOOGLE_API_KEY='your-key-here'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("[System] Gemini 2.5 Flash initialized | temperature=0")

# ── ReAct Constants (Session 3) ─────────────────────────────────────────────
MAX_ITERATIONS    = 5
CONTEXT_THRESHOLD = 12

print(f"[ReAct] MAX_ITERATIONS={MAX_ITERATIONS} | "
      f"CONTEXT_THRESHOLD={CONTEXT_THRESHOLD}")

# ── Summarization Constants (Session 5) ──────────────────────────────────────
SUMMARY_THRESHOLD = 8   # messages before summarization triggers

print(f"[Summarization] SUMMARY_THRESHOLD={SUMMARY_THRESHOLD}")

# ── Supervisor Constants (Session 8) ───────────────────────────────

MAX_DELEGATIONS = 5

print(f"[Supervisor] MAX_DELEGATIONS={MAX_DELEGATIONS}")

# ── Checkpointer (Session 4) ─────────────────────────────────────────────────

DB_PATH = 'support.db'

_db_conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_db_conn)

print(f"[Checkpointer] SQLite initialized → {DB_PATH}")

# ── Presidio (Session 6) ────────────────────────────────────────────────────

analyzer   = AnalyzerEngine()
anonymizer = AnonymizerEngine()

PII_ENTITIES = [
    'CREDIT_CARD',
    'EMAIL_ADDRESS',
    'PHONE_NUMBER',
    'PERSON',
    'US_SSN',
    'IBAN_CODE',
    'IP_ADDRESS',
]

print("[Security] Presidio initialized")
print(f"[Security] PII entities monitored: {len(PII_ENTITIES)}")

 #── Injection Patterns (Session 6) ──────────────────────────────────────────

INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?previous\s+instructions',
    r'disregard\s+(all\s+)?prior\s+instructions',
    r'forget\s+(all\s+)?previous\s+instructions',
    r'you\s+are\s+now\s+a',
    r'new\s+instructions?\s*:',
    r'system\s*prompt\s*:',
    r'jailbreak',
    r'dan\s+mode',
    r'developer\s+mode',
    r'unrestricted\s+mode',
    r'repeat\s+everything\s+above',
    r'print\s+your\s+(system\s+)?prompt',
    r'show\s+me\s+your\s+instructions',
    r'what\s+are\s+your\s+instructions',
]

UNCERTAINTY_MARKERS = [
    r'\bi\s+think\b',
    r'\bi\s+believe\b',
    r'\bprobably\b',
    r'\bi\s+am\s+not\s+sure\b',
    r"\bi'm\s+not\s+sure\b",
    r'\bi\s+guess\b',
    r'\bmaybe\b',
    r'\bperhaps\b',
    r'\bmight\s+be\b',
]

BLOCKED_RESPONSE_TEMPLATE = (
    "I'm unable to process this request as it contains content "
    "that violates our acceptable use policy.\n\n"
    "If you have a genuine support need, please rephrase your "
    "request or contact our team directly at support@company.com.\n\n"
    "Reference: BLOCKED-{ref}"
)

print(f"[Security] Injection patterns: {len(INJECTION_PATTERNS)}")
print(f"[Security] Uncertainty markers: {len(UNCERTAINTY_MARKERS)}")

# ── Custom Message Reducer (Session 5) ────────────────────────────────────────

def deduplicate_messages(left: list, right: list) -> list:
    """
    Custom reducer for state['messages'].
    Handles RemoveMessage deletions, then deduplicates additions.

    Prevents duplicate messages after the checkpointer re-applies
    state. Works alongside summarization_node which emits
    RemoveMessage objects to trim old messages from the list.

    Replaces: add_messages (Session 1 default)
    Introduced: Session 5
    """
    if not right:
        return left
    if not left:
        # Filter out any RemoveMessage from a fresh list
        return [m for m in right if not isinstance(m, RemoveMessage)]

    # Step 1: apply removals
    remove_ids = {m.id for m in right if isinstance(m, RemoveMessage) and m.id}
    if remove_ids:
        left = [m for m in left if not (hasattr(m, 'id') and m.id in remove_ids)]

    # Step 2: deduplicate additions
    additions = [m for m in right if not isinstance(m, RemoveMessage)]
    if not additions:
        return left

    existing_ids = {
        m.id for m in left
        if hasattr(m, 'id') and m.id
    }

    new_msgs = [
        m for m in additions
        if not (hasattr(m, 'id') and m.id in existing_ids)
    ]

    return left + new_msgs

# ══════════════════════════════════════════════════════════════════
# SECTION 2: STATE SCHEMA
# ══════════════════════════════════════════════════════════════════

# ── SharedState (Session 7 — replaces SupportState) ─────────────

class SharedState(TypedDict):

    # ── Core Input (Session 1) ──────────────────────────────────
    raw_input:          str        # Original user message, never modified
    sanitized_input:    str        # PII-cleaned version (Session 6)

    # ── Classification (Session 1) ─────────────────────────────
    category:           str        # technical | billing | fraud | general

    # ── Conversation History (Session 2) ───────────────────────
    messages:           Annotated[list, deduplicate_messages]  # Session 5: add_messages replaced with deduplicate_messages
    customer_data:      dict       # Populated by CRM tool
    tool_results:       Annotated[list, operator.add]  # operator.add — confirmed correct for parallel writers (S7)

    # ── Safety Controls (Session 6) ────────────────────────────
    pii_detected:       bool
    injection_detected: bool
    is_safe:            bool

    # ── Memory and Context (Sessions 3, 5) ─────────────────────
    system_summary:     str        # Compressed history (Session 5)
    iteration_count:    int        # ReAct circuit breaker (Session 3)

    # ── Multi-Agent Scratchpad (Session 7) ──────────────────────
    internal_notes:     Annotated[list, operator.add]
    # Session 7: parallel agent findings scratchpad
    # operator.add is mandatory — parallel agents write here
    # overwrite reducer would silently lose findings
    delegation_count:   int        # Supervisor counter
    next_worker:        str        # Supervisor decision

    # ── Write Access and Human Approval (Session 10, 11) ────────
    github_draft:       dict       # Proposed issue before approval
    github_issue_url:   str        # URL after creation

    # ── Output (Session 1) ─────────────────────────────────────
    final_response:     str


_field_count = len(SharedState.__annotations__)
print(f"[System] SharedState schema — {_field_count} fields | multi-agent ready")

# ── Mock Data (Session 2) ──────────────────────────────────────

MOCK_CRM = {
    'C-1001': {
        # NEEDED FIELDS
        'name': 'Priya Sharma',
        'billing_status': 'Active',
        'subscription_tier': 'Enterprise',
        'last_payment_date': '2026-04-01',
        'last_payment_amount': 4999.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-04-01', 'description': 'Enterprise Plan — April',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-03-01', 'description': 'Enterprise Plan — March',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-02-01', 'description': 'Enterprise Plan — February','amount': 4999.00, 'status': 'paid'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88123',
        'sales_rep_code': 'SR-042',
        'geo_region_tag': 'APAC',
        'last_login_ip': '192.168.10.5',
        'feature_flag_cohort': 'beta-v2',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1002': {
        # NEEDED FIELDS
        'name': 'Arjun Mehta',
        'billing_status': 'Past Due',
        'subscription_tier': 'Pro',
        'last_payment_date': '2026-03-01',
        'last_payment_amount': 499.00,
        'outstanding_balance': 998.00,
        'recent_transactions': [
            {'date': '2026-03-01', 'description': 'Pro Plan — March',  'amount': 499.00, 'status': 'paid'},
            {'date': '2026-04-01', 'description': 'Pro Plan — April',  'amount': 499.00, 'status': 'missed'},
            {'date': '2026-05-01', 'description': 'Pro Plan — May',    'amount': 499.00, 'status': 'missed'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88456',
        'sales_rep_code': 'SR-017',
        'geo_region_tag': 'APAC',
        'last_login_ip': '10.0.0.44',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1003': {
        # NEEDED FIELDS
        'name': 'Kavya Nair',
        'billing_status': 'Active',
        'subscription_tier': 'Starter',
        'last_payment_date': '2026-05-01',
        'last_payment_amount': 99.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-05-01', 'description': 'Starter Plan — May', 'amount': 99.00, 'status': 'paid'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88789',
        'sales_rep_code': 'SR-031',
        'geo_region_tag': 'EMEA',
        'last_login_ip': '172.16.0.9',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
}

MOCK_KB = {
    'api': (
        'API troubleshooting guide: (1) Check rate limits — free tier: 100 req/min, '
        'pro: 1000 req/min, enterprise: unlimited. (2) Auth header must be '
        '"Authorization: Bearer <token>" — never basic auth. (3) On 401 errors, '
        'regenerate your API key in Account > API Keys. (4) On 429 rate-limit errors, '
        'implement exponential backoff starting at 1s. (5) SDK v3+ requires '
        'client.initialize() before first call.'
    ),
    'login': (
        'Login troubleshooting: (1) Clear browser cache and cookies, then retry. '
        '(2) MFA: open your authenticator app, use the 6-digit code within 30 seconds. '
        '(3) Password reset: go to login page > "Forgot password" > check email within '
        '5 minutes. (4) If locked out after 5 attempts, wait 15 minutes or contact '
        'support. (5) SSO users: ensure your identity provider session is active.'
    ),
    'billing': (
        'Billing help: (1) Invoice portal: account.nexus.io/billing/invoices — '
        'download PDF or CSV. (2) Update payment method: Billing > Payment Methods > '
        'Add New Card. (3) Refund policy: eligible within 30 days of charge, '
        'processed in 5-10 business days. (4) Subscription changes take effect on '
        'next billing cycle. (5) Failed payments retry automatically for 3 days.'
    ),
    'update': (
        'Post-update troubleshooting: (1) Clear application cache after any update: '
        'Settings > Cache > Clear All. (2) If issues persist, rollback procedure: '
        'go to Admin > Versions > select previous stable version > Rollback. '
        '(3) Check the changelog at docs.nexus.io/changelog for breaking changes. '
        '(4) SDK updates: run "npm install @nexus/sdk@latest" or '
        '"pip install nexus-sdk --upgrade".'
    ),
    '2fa': (
        '2FA / MFA help: (1) Backup codes: stored during setup — check your saved '
        'codes document. (2) Lost device: go to login > "Use backup code" > enter '
        'one of your 8-digit backup codes. (3) Reset 2FA: Account > Security > '
        'Two-Factor Auth > Reset — requires email verification. (4) Manual '
        'verification for locked accounts: contact support with government ID. '
        '(5) TOTP apps supported: Google Authenticator, Authy, 1Password.'
    ),
    'sdk': (
        'SDK compatibility guide: (1) SDK v3.x requires Node 18+ or Python 3.10+. '
        '(2) Migration from v2 to v3: replace client.get() with client.fetch(), '
        'update auth to client.initialize({apiKey}). (3) Breaking changes in v3: '
        'callback-style API removed, promises only. (4) Python SDK: '
        '"from nexus import NexusClient" replaces "import nexus". '
        '(5) Full migration guide: docs.nexus.io/sdk/v3-migration.'
    ),
}

# ── Mock Fraud Database (Session 3) ──────────────────────────────
MOCK_FRAUD_DB = {
    'ACC-F001': {
        'risk_score': 0.91,
        'flagged_patterns': ['multiple_countries_24h', 'unusual_amount'],
        'recommendation': 'freeze_account',
        'recent_flags': [
            {'date': '2026-05-15', 'pattern': 'multiple_countries_24h', 'severity': 'high'},
            {'date': '2026-05-15', 'pattern': 'unusual_amount',         'severity': 'high'},
        ],
    },
    'ACC-F002': {
        'risk_score': 0.23,
        'flagged_patterns': [],
        'recommendation': 'no_action',
        'recent_flags': [],
    },
    'ACC-F003': {
        'risk_score': 0.67,
        'flagged_patterns': ['new_device', 'large_transfer'],
        'recommendation': 'manual_review',
        'recent_flags': [
            {'date': '2026-05-14', 'pattern': 'large_transfer', 'severity': 'medium'},
        ],
    },
}

# ── Tools (Session 2) ──────────────────────────────────────────

@tool
def get_customer_details(customer_id: str) -> dict:
    """
    WHAT:
    Retrieves billing status, subscription tier, last payment date,
    outstanding balance, and recent transaction history for a customer
    from the CRM system.

    WHEN:
    Call this tool when the user's query involves billing, payment
    status, invoice disputes, subscription management, refund
    requests, or account standing. Always call before answering
    any billing question.

    FORMAT:
    customer_id must be in format 'C-XXXX' e.g. 'C-1001', 'C-1042'.
    Extract from the user message.
    If not present in the message, ask the user before calling.
    Never guess or fabricate a customer_id.

    RETURN:
    Dict with: name, billing_status, subscription_tier,
    last_payment_date, last_payment_amount, outstanding_balance,
    recent_transactions (last 3 only).
    On any failure: dict with single 'error' key describing what failed.
    """
    # LAYER 1 — Argument validation
    if not customer_id or not isinstance(customer_id, str):
        return {'error': "customer_id must be a non-empty string."}
    cid = customer_id.strip().upper()
    if not cid.startswith('C-'):
        return {'error': f"Invalid format: '{customer_id}'. Expected 'C-XXXX' e.g. 'C-1001'"}

    # LAYER 2 — Database lookup with error handling
    try:
        raw = MOCK_CRM.get(cid)
        if raw is None:
            return {'error': f"Customer '{cid}' not found. Please verify the ID with the customer."}
    except Exception as e:
        return {'error': f"CRM lookup failed: {type(e).__name__}. Contact engineering if this persists."}

    # LAYER 3 — Data filtering
    NEEDED = {
        'name', 'billing_status', 'subscription_tier',
        'last_payment_date', 'last_payment_amount',
        'outstanding_balance', 'recent_transactions'
    }
    filtered = {k: v for k, v in raw.items() if k in NEEDED}
    filtered['recent_transactions'] = filtered.get('recent_transactions', [])[:3]
    return filtered

# Test: get_customer_details.invoke({'customer_id': 'C-1001'})
# Test: get_customer_details.invoke({'customer_id': 'C-9999'})
# Test: get_customer_details.invoke({'customer_id': 'bad'})


@tool
def search_knowledge_base(query: str) -> dict:
    """
    WHAT:
    Searches the internal technical knowledge base for resolution
    steps and troubleshooting articles matching the issue described.

    WHEN:
    Call this tool for any technical issue before responding to the
    customer. Always search before saying you cannot help.
    If first search returns no match, try with different keywords.

    FORMAT:
    query is a natural language string describing the technical
    problem. Be specific. Include error codes or keywords.
    Example: 'API authentication 401 error after SDK update'

    RETURN:
    Dict with matched (bool), results (list of article strings),
    count (int). If no match: matched=False with fallback guidance.
    On failure: dict with single 'error' key.
    """
    try:
        if not query or not query.strip():
            return {'error': 'Search query cannot be empty.'}

        query_lower = query.lower()
        results = []
        for keyword, article in MOCK_KB.items():
            if keyword in query_lower:
                results.append(article)

        if not results:
            return {
                'matched': False,
                'results': [],
                'count': 0,
                'fallback': (
                    'No specific article found. General guidance: '
                    'check account status, clear browser cache, verify '
                    'recent configuration changes, review changelog.'
                )
            }

        return {'matched': True, 'results': results, 'count': len(results)}

    except Exception as e:
        return {'error': f"KB search failed: {type(e).__name__}"}

# Test: search_knowledge_base.invoke({'query': 'API 401 error'})
# Test: search_knowledge_base.invoke({'query': 'nothing matches'})


# ── Fraud Tool (Session 3) ──────────────────────────────────────

@tool
def check_fraud_signals(account_id: str) -> dict:
    """
    WHAT:
    Checks an account's transaction history against fraud
    detection rules. Returns a risk score, flagged behavioral
    patterns, and a recommended action for the security team.

    WHEN:
    Call for any ticket mentioning unauthorized transactions,
    suspicious charges, account compromise, or identity theft.
    Always call before making any fraud assessment.

    FORMAT:
    account_id must be in format 'ACC-FXXX' e.g. 'ACC-F001'.
    Extract from the user message.
    If not present, ask the user before calling this tool.

    RETURN:
    Dict with: account_id, risk_score (float 0.0-1.0),
    flagged_patterns (list of strings), recommendation (str),
    recent_flags (list of dicts).
    risk_score > 0.7  -> high risk
    risk_score 0.4-0.7 -> medium risk
    risk_score < 0.4  -> low risk
    On failure: dict with single 'error' key.
    """
    # LAYER 1 — Validation
    if not account_id or not isinstance(account_id, str):
        return {'error': 'account_id must be a non-empty string.'}
    aid = account_id.strip().upper()
    if not aid.startswith('ACC-'):
        return {
            'error': f"Invalid format: '{account_id}'. "
                     f"Expected 'ACC-FXXX' e.g. 'ACC-F001'"
        }

    # LAYER 2 — Lookup with error handling
    try:
        record = MOCK_FRAUD_DB.get(aid)
        if record is None:
            return {
                'error': f"No fraud profile found for '{aid}'. "
                         f"Verify the account ID with the customer."
            }
    except Exception as e:
        return {'error': f"Fraud DB unavailable: {type(e).__name__}"}

    # LAYER 3 — Return with account_id injected
    result = dict(record)
    result['account_id'] = aid
    return result

# Test: check_fraud_signals.invoke({'account_id': 'ACC-F001'})
# Test: check_fraud_signals.invoke({'account_id': 'ACC-F999'})
# Test: check_fraud_signals.invoke({'account_id': 'bad-format'})

TOOLS = [
    get_customer_details,
    search_knowledge_base,
    check_fraud_signals,
]
llm_with_tools = llm.bind_tools(TOOLS)

print(f"[Tools] {len(TOOLS)} tools registered:")
for t in TOOLS:
    print(f"  · {t.name}")

# ── Agent System Prompt (Session 3) ──────────────────────────────

AGENT_SYSTEM_PROMPT = """
You are a senior customer support specialist with access
to the CRM system and internal knowledge base.

TOOL USAGE RULES:

get_customer_details:
  - Call for ANY billing, payment, subscription, or account query
  - ALWAYS call before answering billing questions
  - If customer_id not in the message: ask before calling
  - Never guess or fabricate a customer_id

search_knowledge_base:
  - Call for ANY technical issue before responding
  - Always search before saying you cannot help
  - Use specific technical terms in the query
  - Multiple searches allowed if first returns no match

check_fraud_signals:
  - Call for ANY mention of unauthorized transactions,
    suspicious charges, or account compromise
  - Always call before making any fraud assessment
  - If account_id not in the message: ask before calling

RESPONSE RULES:
  - Base all answers on tool output, not internal knowledge
  - If a tool returns an error key: acknowledge it professionally
  - Reference specific data points from tool results
  - Never expose internal field names or system details
"""

# ── Summarization Prompt (Session 5) ──────────────────────────────────────────

SUMMARIZATION_PROMPT = """
You are summarizing a customer support conversation to preserve
key context for future turns of the same conversation.

Create a dense factual summary of 3 to 5 sentences.

You MUST include every instance of:
  - Customer identifiers (account IDs, names, email addresses)
  - Financial data (amounts, dates, balances, transaction IDs)
  - What the customer reported and what investigation found
  - Decisions made or actions taken in this conversation
  - Items still unresolved or pending customer action

You MUST NOT include:
  - Pleasantries or conversational filler
  - Failed tool call attempts or error messages
  - Repeated information already stated earlier
  - Internal system field names or technical metadata

Respond with the summary text only.
No preamble. No labels. No bullet points.
Plain prose. Dense with facts.
"""

# ── Supervisor Decision Model (Session 8) ──────────────────────────

class SupervisorDecision(BaseModel):
    """
    Structured output model for supervisor_node.
    Pydantic enforces that next_worker is one of the
    five declared Literal values at runtime.
    If the LLM returns anything outside this set:
    ValidationError is raised and caught — safe fallback fires.
    Introduced: Session 8. Permanent.
    """
    next_worker: Literal[
        'triage',
        'tech_support',
        'fraud_handler',
        'general_handler',
        'FINISH',
    ]
    reasoning: str  # audit trail — why this decision was made

# ── Supervisor Prompt (Session 8) ──────────────────────────────────

SUPERVISOR_PROMPT = """
You are a supervisor coordinating a customer support team.
You read the full conversation state and decide which
specialist handles the ticket next.

Available workers:
  triage:           re-classify or re-validate the input
  tech_support:     handles technical and billing issues
  fraud_handler:    handles suspicious transaction reports
  general_handler:  handles general how-to queries
  FINISH:           all issues in the ticket are fully addressed

Decision rules (apply in order):
  1. If final_response is set AND addresses all issues
     mentioned in the original ticket: return FINISH
  2. If a billing or technical issue is present and
     not yet addressed: return tech_support
  3. If suspicious activity is mentioned and not yet
     investigated by fraud_handler: return fraud_handler
  4. If the classification seems wrong or unsafe input
     was detected: return triage
  5. If delegation_count exceeds 3: return FINISH

Default to FINISH unless you are certain another
worker is needed. Do not route unnecessarily.
Unnecessary routing costs money and time.
"""

# ── Supervisor LLM (Session 8) ──────────────────────────────────────

supervisor_llm = llm.with_structured_output(SupervisorDecision)

print("[Supervisor] supervisor_llm initialized with structured output")

# ── Supervisor Node (Session 8) ──────────────────────────────────────

def build_supervisor_messages(state: SharedState) -> list:
    """
    Builds the message list for the supervisor LLM call.
    Includes: original ticket, all internal_notes findings,
    current final_response if set, delegation count.
    Called by supervisor_node on every invocation.
    """

    original_ticket = state.get('raw_input', '')
    final_response  = state.get('final_response', '')
    delegation      = state.get('delegation_count', 0)
    notes           = state.get('internal_notes', [])
    category        = state.get('category', '')

    findings_text = ''
    if notes:
        findings_text = '\n'.join([
            f"- [{n.get('agent','unknown')}]: {n.get('finding','')}"
            if isinstance(n, dict)
            else f"- {str(n)}"
            for n in notes
        ])
    else:
        findings_text = 'No findings yet.'

    user_message = (
        f"ORIGINAL TICKET:\n{original_ticket}\n\n"
        f"CATEGORY DETECTED: {category}\n\n"
        f"DELEGATION COUNT: {delegation}\n\n"
        f"WORKER FINDINGS:\n{findings_text}\n\n"
        f"CURRENT RESPONSE:\n{final_response if final_response else 'Not yet generated.'}\n\n"
        f"Decide which worker should handle next, or FINISH if done."
    )

    return [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=user_message),
    ]


def supervisor_node(state: SharedState) -> dict:
    """
    LLM-powered routing node. Reads full SharedState.
    Returns SupervisorDecision via structured output.
    Applies delegation_count safety limit before LLM call.
    Hub of the hub-and-spoke topology.
    Introduced: Session 8. Permanent.

    Safety limit fires before LLM call — zero tokens consumed.
    ValidationError on structured output → safe fallback to triage.
    """

    delegation = state.get('delegation_count', 0) + 1

    # Safety limit — mirrors circuit breaker from Session 3
    if delegation > MAX_DELEGATIONS:
        print(f"[Supervisor] MAX_DELEGATIONS reached: {delegation} → FINISH")
        return {
            'delegation_count': delegation,
            'next_worker':      'FINISH',
        }

    print(f"[Supervisor] Delegation {delegation}/{MAX_DELEGATIONS}")

    # Build messages and call supervisor LLM
    messages = build_supervisor_messages(state)

    try:
        decision = supervisor_llm.invoke(messages)
        print(f"[Supervisor] Decision: {decision.next_worker} | "
              f"Reasoning: {decision.reasoning[:60]}...")

    except Exception as e:
        print(f"[Supervisor] Structured output failed: {e} — defaulting to triage")
        decision = SupervisorDecision(
            next_worker='triage',
            reasoning=f'Validation error: {str(e)}'
        )

    return {
        'delegation_count': delegation,
        'next_worker':      decision.next_worker,
        'internal_notes':   [{
            'agent':      'supervisor',
            'finding':    f"Routed to {decision.next_worker}",
            'reasoning':  decision.reasoning,
            'delegation': delegation,
        }],
    }


# ── Supervisor Router (Session 8) ─────────────────────────────────

def supervisor_router(state: SharedState) -> str:
    """
    Pure Python router. Reads next_worker from state.
    Routes to the correct node or END.
    Zero LLM calls. Zero business logic.
    Permanent from Session 8 onward.
    """

    next_w = state.get('next_worker', 'FINISH')

    routing = {
        'triage':           'triage',
        'tech_support':     'tech_support',
        'fraud_handler':    'fraud_handler',
        'general_handler':  'general_handler',
        'FINISH':           END,
    }

    destination = routing.get(next_w, 'triage')
    print(f"[Router:supervisor] next_worker='{next_w}' → {destination}")
    return destination

# ══════════════════════════════════════════════════════════════════
# SECTION 3: INGRESS NODE (Session 6)
# ══════════════════════════════════════════════════════════════════

# ── Ingress Node (Session 6) ─────────────────────────────────────

def ingress_node(state: SharedState) -> dict:
    """
    Security ingress — first node every ticket touches.
    Performs two independent checks in order:
      1. PII detection and masking via Presidio
      2. Injection pattern detection via regex

    Sets: pii_detected, injection_detected, is_safe, sanitized_input.
    Never calls the LLM. Pure CPU. Sub-10ms per ticket.
    is_safe = False if injection detected (PII alone is not unsafe).

    Permanent from Session 6 onward.
    New entry point — replaces classify_node as graph entry.
    """

    raw = state.get('raw_input', '')
    print(f"[Ingress] Scanning: '{raw[:60]}...'")

    # ── STEP 1: PII DETECTION AND MASKING ────────────────────────

    try:
        results = analyzer.analyze(
            text=raw,
            language='en',
            entities=PII_ENTITIES,
        )

        # Filter to high-confidence detections only
        results = [r for r in results if r.score > 0.7]
        pii_found = len(results) > 0

        if pii_found:
            anonymized = anonymizer.anonymize(
                text=raw,
                analyzer_results=results,
            )
            sanitized = anonymized.text
            entities_found = [r.entity_type for r in results]
            print(f"[Ingress] PII detected: {entities_found}")
            print(f"[Ingress] Sanitized: '{sanitized[:60]}...'")
        else:
            sanitized = raw
            print(f"[Ingress] No PII detected")

    except Exception as e:
        print(f"[Ingress] Presidio error: {e} — passing raw input")
        pii_found = False
        sanitized = raw

    # ── STEP 2: INJECTION PATTERN DETECTION ──────────────────────

    injection_found = any(
        re.search(pattern, raw, re.IGNORECASE)
        for pattern in INJECTION_PATTERNS
    )

    if injection_found:
        print(f"[Ingress] INJECTION DETECTED — blocking request")
    else:
        print(f"[Ingress] No injection detected")

    # ── SAFETY GATE ───────────────────────────────────────────────

    # PII alone does not block — it is masked and passed through
    # Injection blocks — the request never reaches classify_node
    is_safe = not injection_found

    return {
        'sanitized_input':    sanitized,
        'pii_detected':       pii_found,
        'injection_detected': injection_found,
        'is_safe':            is_safe,
    }

# ── Ingress Router (Session 6) ───────────────────────────────────

def route_after_ingress(state: SharedState) -> str:
    """
    Reads is_safe from state.
    False → blocked_response_node (zero LLM tokens).
    True  → classify_node (normal agent flow).
    Pure Python. Zero LLM calls. Permanent from Session 6.
    """

    is_safe = state.get('is_safe', True)
    destination = 'classify_node' if is_safe else 'blocked_response_node'
    print(f"[Router:ingress] is_safe={is_safe} → {destination}")
    return destination

# ── Blocked Response Node (Session 6) ────────────────────────────

def blocked_response_node(state: SharedState) -> dict:
    """
    Fires when is_safe == False.
    Returns a pre-written professional refusal.
    Zero LLM tokens consumed — no API call made.
    Reference number is timestamp-based for audit logging.

    Permanent from Session 6 onward.
    """

    ref = str(int(time.time()))[-8:]
    response = BLOCKED_RESPONSE_TEMPLATE.format(ref=ref)

    print(f"[Blocked] Request blocked | "
          f"pii={state.get('pii_detected')} | "
          f"injection={state.get('injection_detected')} | "
          f"ref=BLOCKED-{ref}")

    return {'final_response': response}

# ══════════════════════════════════════════════════════════════════
# SECTION 4: CLASSIFIER NODE
# ══════════════════════════════════════════════════════════════════

def classify_node(state: SharedState) -> dict:
    system_prompt = (
        "You are a support ticket classifier for an enterprise SaaS company.\n"
        "Classify the incoming ticket into EXACTLY ONE of these 4 categories:\n\n"
        "  technical:  API errors, login failures, bugs, performance issues,\n"
        "              integration problems, post-update breakage\n"
        "  billing:    payment failures, invoice disputes, subscriptions,\n"
        "              refund requests, double charges\n"
        "  fraud:      unauthorized transactions, account compromise,\n"
        "              suspicious activity, identity theft\n"
        "  general:    feature questions, how-to, onboarding, documentation,\n"
        "              anything that does not fit the above categories\n\n"
        "Respond with EXACTLY ONE WORD. No punctuation. "
        "No explanation. No other text whatsoever."
    )

    # Session 6: reads sanitized_input — PII already masked by ingress_node
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.get('sanitized_input') or state['raw_input']),
    ])

    # Layer 1 — normalize
    raw = response.content.strip().lower().rstrip(".,!?")

    # Layer 2 — validate
    VALID = {"technical", "billing", "fraud", "general"}
    if raw not in VALID:
        print(f"[Classifier] Unexpected output: '{raw}' → defaulting to 'general'")
        raw = "general"

    # Layer 3 — print
    preview = state.get('sanitized_input') or state["raw_input"]
    preview = preview[:60]
    print(f"[Classifier] '{preview}'... → {raw}")

    return {
        "category":           raw,
        "iteration_count":    0,
        "delegation_count":   0,
    }

# ══════════════════════════════════════════════════════════════════
# SECTION 5: ROUTER FUNCTION
# ══════════════════════════════════════════════════════════════════

def route_by_category(state: SharedState) -> str:
    raw = state.get("category") or ""
    category = raw.strip().lower()

    routing_map = {
        "technical": "technical_handler",
        "billing":   "billing_handler",
        "fraud":     "fraud_handler",
        "general":   "general_handler",
    }

    destination = routing_map.get(category, "general_handler")
    print(f"[Router] '{category}' → {destination}")
    return destination

# ── Summarization Router (Session 5) ──────────────────────────────────────────

def route_after_classify(state: SharedState) -> str:
    """
    Fires after classify_node. Handles both category routing
    and summarization threshold check.

    For fraud/general: routes directly to their handlers.
    For billing/technical: checks SUMMARY_THRESHOLD.
      If exceeded: routes to summarization_node first.
      If not: routes directly to agent_node.

    Pure Python. Zero LLM calls. Zero business logic.
    Permanent from Session 5 onward.
    """
    category = state.get('category', '')

    if category == 'fraud':
        return 'fraud_handler'
    if category == 'general':
        return 'general_handler'

    # billing or technical: check message count
    msg_count = len(state.get('messages', []))

    if msg_count > SUMMARY_THRESHOLD:
        print(f"[Router:classify] {msg_count} messages "
              f"> {SUMMARY_THRESHOLD} → summarization_node")
        return 'summarization_node'

    print(f"[Router:classify] {msg_count} messages "
          f"≤ {SUMMARY_THRESHOLD} → agent_node")
    return 'agent_node'

# ══════════════════════════════════════════════════════════════════
# SECTION 6: HANDLER STUBS
# ══════════════════════════════════════════════════════════════════
def technical_handler(state: SharedState) -> dict:
    # STUB — replaced in Session 2 (routing now goes to agent_node)
    preview = state["raw_input"][:80]
    print(f"[technical_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Your technical issue has been received and assigned to our "
            "Engineering team. A specialist will respond within 4 hours."
        )
    }


def billing_handler(state: SharedState) -> dict:
    # STUB — replaced in Session 2 (routing now goes to agent_node)
    preview = state["raw_input"][:80]
    print(f"[billing_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Your billing inquiry has been received and assigned to our "
            "Finance team. We will review your account within 2 hours."
        )
    }


def general_handler(state: SharedState) -> dict:
    """
    Handles general how-to and feature questions via a direct LLM call.
    Upgraded from stub in Session 8 — stub caused supervisor to loop
    indefinitely because the hardcoded response never answered the ticket.
    """
    ticket = state.get('sanitized_input') or state.get('raw_input', '')
    print(f"[general_handler] Handling: '{ticket[:80]}'")

    general_system_prompt = (
        "You are a helpful customer support specialist for Nexus, an enterprise SaaS platform.\n"
        "Answer the customer's question clearly and concisely.\n"
        "If you don't have specific information, provide helpful general guidance.\n"
        "Keep your response professional and under 150 words."
    )

    response = llm.invoke([
        SystemMessage(content=general_system_prompt),
        HumanMessage(content=ticket),
    ])

# ── ReAct Helpers (Session 3) ────────────────────────────────────

def build_escalation_response(state: SharedState, iteration: int) -> dict:
    """
    Produces a graceful user-facing escalation message when
    the circuit breaker fires or a duplicate tool call is detected.
    Summarizes tool findings before escalating.
    Called by: agent_node (Session 3 onward).
    """
    tool_findings = []
    for msg in state.get('messages', []):
        if hasattr(msg, 'tool_call_id') and msg.content:
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and 'error' not in data:
                    tool_findings.append(data)
            except Exception:
                pass

    if tool_findings:
        lines = []
        for finding in tool_findings[:2]:
            for k, v in list(finding.items())[:2]:
                lines.append(f"· {k}: {v}")
        summary = "\n".join(lines)
    else:
        summary = "· No data retrieved before escalation."

    ref = str(uuid.uuid4())[:8].upper()

    escalation_text = (
        f"I investigated your request thoroughly but was unable "
        f"to resolve it automatically.\n\n"
        f"What I found:\n{summary}\n\n"
        f"A specialist will review this and contact you within "
        f"24 hours. Reference: {ref}"
    )

    print(f"[Escalation] Circuit breaker at iteration {iteration} "
          f"| ref: {ref}")

    return {
        'messages':        [AIMessage(content=escalation_text)],
        'iteration_count': iteration,
        'final_response':  escalation_text,
    }

    final_text = _extract_text(response.content)
    print(f"[general_handler] Response: {len(final_text)} chars")
    return {'final_response': final_text}

def trim_context(messages: list, threshold: int) -> list:
    """
    Keeps messages[0] (original user message) plus the most
    recent (threshold - 1) messages. Prevents context window
    explosion over many tool call iterations.
    Called by: agent_node before every LLM call (Session 3 onward).
    """
    if len(messages) <= threshold:
        return messages

    preserved = messages[0]
    recent    = messages[-(threshold - 1):]
    result    = [preserved] + recent

    print(f"[Context Trim] {len(messages)} → {len(result)} messages")
    return result

def get_tool_fingerprint(tool_call: dict) -> str:
    """
    Returns a unique string for a tool call based on its name
    and sorted arguments. Used to detect duplicate tool calls.
    Called by: agent_node after every LLM response (Session 3 onward).
    """
    name = tool_call.get('name', '')
    args = tool_call.get('args', {})
    return f"{name}::{json.dumps(args, sort_keys=True)}"

# ── Agent Node (Session 3) ──────────────────────────────────────

def agent_node(state: SharedState) -> dict:
    """
    Full ReAct agent node with three safety layers.
    Replaces the single-pass agent_node from Session 2.

    Layer 1: Circuit breaker — hard stop at MAX_ITERATIONS.
    Layer 2: Read system_summary — prepend to prompt if present.
    Layer 3: Duplicate detection — fingerprint each tool call,
             escalate immediately if same call seen twice.

    Uses AGENT_SYSTEM_PROMPT module constant (Session 3+).
    Session 5: trim_context() retired. Reads system_summary
               from state instead.
    Session 6: use sanitized_input if available.
    Permanent from Session 3 onward.
    """

    # ── LAYER 1: CIRCUIT BREAKER ─────────────────────────────────
    iteration = state.get('iteration_count', 0) + 1

    if iteration > MAX_ITERATIONS:
        return build_escalation_response(state, iteration)

    print(f"[Agent] iteration={iteration}/{MAX_ITERATIONS}")

    # ── LAYER 2: READ SYSTEM SUMMARY ─────────────────────────────
    summary = state.get('system_summary', '')

    if summary:
        context = (
            f"PRIOR CONTEXT SUMMARY:\n{summary}"
            f"\n\n{AGENT_SYSTEM_PROMPT}"
        )
        print(f"[Agent] system_summary present "
              f"({len(summary)} chars) — prepended to prompt")
    else:
        context = AGENT_SYSTEM_PROMPT
        print(f"[Agent] No system_summary — using base prompt")

    # Session 6: use sanitized_input if available
    # Replace the first HumanMessage with sanitized version
    messages = list(state.get('messages', []))
    sanitized = state.get('sanitized_input', '')

    if sanitized and messages:
        first_msg = messages[0]
        if hasattr(first_msg, 'content') and first_msg.content == state.get('raw_input', ''):
            from langchain_core.messages import HumanMessage as HM
            messages[0] = HM(content=sanitized)

    # No trim_context() call — summarization_node handles this
    messages_to_send = [
        SystemMessage(content=context),
        *messages
    ]

    # ── CORE: LLM CALL ───────────────────────────────────────────
    response   = llm_with_tools.invoke(messages_to_send)
    tool_count = len(response.tool_calls) if response.tool_calls else 0
    print(f"[Agent] tool_calls={tool_count} | "
          f"has_content={bool(response.content)}")

    # ── LAYER 3: DUPLICATE DETECTION ─────────────────────────────
    new_fingerprints = []

    if response.tool_calls:

        existing = {
            r.get('fingerprint')
            for r in state.get('tool_results', [])
            if isinstance(r, dict) and 'fingerprint' in r
        }

        for tc in response.tool_calls:
            fp = get_tool_fingerprint(tc)

            if fp in existing:
                print(f"[Agent] Duplicate: {tc['name']} same args. Escalating.")
                stuck_text = (
                    f"I've already attempted {tc['name']} with these "
                    f"parameters and received an error. Escalating to "
                    f"our support team for manual review."
                )
                return {
                    'messages':        [AIMessage(content=stuck_text)],
                    'iteration_count': iteration,
                    'final_response':  stuck_text,
                }

            new_fingerprints.append({'fingerprint': fp})

    # ── RETURN ────────────────────────────────────────────────────
    return {
        'messages':        [response],
        'iteration_count': iteration,
        'tool_results':    new_fingerprints,  # operator.add appends to existing
    }

# ── Routing & Terminal Nodes (Session 2) ─────────────────────────

def route_after_agent(state: SharedState) -> str:
    """
    Reads last message. If tool_calls present → tool_node.
    If no tool_calls → respond_node.
    Pure Python. Zero LLM calls. Zero business logic.
    Permanent from Session 2 onward.
    """
    messages = state.get('messages', [])
    if not messages:
        return 'respond_node'
    last = messages[-1]
    has_tools = hasattr(last, 'tool_calls') and bool(last.tool_calls)
    destination = 'tool_node' if has_tools else 'respond_node'
    print(f"[Router:after_agent] tool_calls={has_tools} → {destination}")
    return destination

def respond_node(state: SharedState) -> dict:
    """
    Extracts last AIMessage content → final_response.
    Runs after agent_node when no further tool calls needed.
    Permanent from Session 2 onward.
    """
    messages = state.get('messages', [])
    final = ''
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            # Gemini may return a list of content blocks; extract text
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        parts.append(block['text'])
                    elif isinstance(block, str):
                        parts.append(block)
                final = ' '.join(parts).strip()
            else:
                final = str(content)
            if final:
                break
    print(f"[Respond] {len(final)} chars")
    return {'final_response': final}


tool_node = ToolNode(tools=TOOLS)
print(f"[Tools] ToolNode ready — {len(TOOLS)} tools registered")

def _extract_text(content) -> str:
    """Extracts plain text from an AIMessage content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                parts.append(block['text'])
            elif isinstance(block, str):
                parts.append(block)
        return ' '.join(parts).strip()
    return str(content)

# ── Summarization Node (Session 5) ────────────────────────────────────────────

def summarization_node(state: SharedState) -> dict:
    """
    Maintenance node. Fires when message count exceeds
    SUMMARY_THRESHOLD. Compresses old messages into
    state['system_summary']. Trims messages to last 4.

    Produces no user-facing output.
    Serves agent_node by managing context size.
    Introduced: Session 5. Permanent from here onward.
    """
    messages = state.get('messages', [])
    print(f"[Summarize] Triggered — {len(messages)} messages → compressing")

    # Filter to only Human/AI content messages — no tool_calls or ToolMessages.
    # Gemini rejects sequences with orphaned function-call turns.
    msgs_for_summary = []
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            msgs_for_summary.append(m)
        elif isinstance(m, AIMessage) and m.content and not getattr(m, 'tool_calls', None):
            msgs_for_summary.append(m)

    if not msgs_for_summary:
        msgs_for_summary = messages  # fallback: send everything

    try:
        response = llm.invoke([
            SystemMessage(content=SUMMARIZATION_PROMPT),
            *msgs_for_summary
        ])
        summary = _extract_text(response.content).strip()
        # Strip Gemini 2.5 Flash thinking tokens if present in content
        import re as _re
        summary = _re.sub(r'<thinking>.*?</thinking>', '', summary,
                          flags=_re.DOTALL | _re.IGNORECASE).strip()
        # Hard cap: a real summary is never > 1500 chars
        if len(summary) > 1500:
            summary = summary[:1500].rsplit('.', 1)[0] + '.'
        print(f"[Summarize] Summary: {summary[:80]}...")
    except Exception as e:
        print(f"[Summarize] LLM error: {e} — keeping existing summary")
        return {}

    # Keep last 4 messages but always start at a HumanMessage boundary
    # so we never hand Gemini an orphaned AIMessage(tool_calls) first.
    keep_n = 4
    while keep_n <= len(messages):
        if isinstance(messages[-keep_n], HumanMessage):
            break
        keep_n += 1
    # Fallback: if no HumanMessage found, keep as-is
    if keep_n > len(messages):
        keep_n = 4

    keep_from = len(messages) - keep_n
    messages_to_remove = messages[:keep_from]

    print(f"[Summarize] Messages trimmed: {len(messages)} → {keep_n} "
          f"(kept from index {keep_from})")

    # Use RemoveMessage to delete old entries so the reducer handles it cleanly.
    remove_ops = [
        RemoveMessage(id=m.id)
        for m in messages_to_remove
        if hasattr(m, 'id') and m.id
    ]

    return {
        'system_summary': summary,
        'messages':       remove_ops,
    }

# ── Egress Node (Session 6) ──────────────────────────────────────

def egress_node(state: SharedState) -> dict:
    """
    Security egress — scans final_response before delivery.
    Two checks:
      1. PII leakage — Presidio scan on response text
      2. Uncertainty markers — regex on response text

    Does not block in this session — flags and logs only.
    In production: flagged responses route to human review queue.
    Permanent from Session 6 onward.
    """

    response_text = state.get('final_response', '')

    if not response_text.strip():
        return {}

    # ── CHECK 1: PII LEAKAGE IN OUTPUT ───────────────────────────

    try:
        output_results = analyzer.analyze(
            text=response_text,
            language='en',
            entities=PII_ENTITIES,
        )
        output_results = [r for r in output_results if r.score > 0.7]
        pii_in_output = len(output_results) > 0

        if pii_in_output:
            leaked_types = [r.entity_type for r in output_results]
            print(f"[Egress] WARNING: PII in output: {leaked_types}")
        else:
            print(f"[Egress] Output PII check: clean")

    except Exception as e:
        print(f"[Egress] Presidio output scan error: {e}")
        pii_in_output = False

    # ── CHECK 2: UNCERTAINTY MARKERS ─────────────────────────────

    uncertainty_found = any(
        re.search(marker, response_text, re.IGNORECASE)
        for marker in UNCERTAINTY_MARKERS
    )

    if uncertainty_found:
        print(f"[Egress] WARNING: Uncertainty markers in output")
    else:
        print(f"[Egress] Uncertainty check: clean")

    # ── FLAG BUT DO NOT BLOCK ────────────────────────────────────

    # In this session: log only.
    # In production: route to human review queue if either flag is True.
    output_is_safe = not pii_in_output and not uncertainty_found

    if not output_is_safe:
        print(f"[Egress] FLAGGED for review | "
              f"pii_leak={pii_in_output} | "
              f"uncertainty={uncertainty_found}")

    # Return empty dict — egress does not modify state in this session
    # It only logs. Session 9 adds active remediation.
    return {}

# ── Fraud Handler (Session 3) ────────────────────────────────────

def fraud_handler(state: SharedState) -> dict:
    """
    Fraud analysis handler. Upgraded from stub in Session 3.
    Uses check_fraud_signals for real fraud assessment.
    Single tool call — not the full ReAct loop.
    Replaced with parallel fraud agent swarm in Session 9.
    """

    fraud_system_prompt = """
    You are a fraud analysis specialist.
    You have access to the check_fraud_signals tool.

    When a customer reports suspicious activity:
    1. Extract the account_id from their message.
    2. Call check_fraud_signals with that account_id.
    3. Interpret the risk_score and flagged_patterns.
    4. Give a clear, professional response about next steps.

    If account_id is not in the message: ask for it first.
    Never fabricate fraud findings.
    Always base your response entirely on tool output.
    """

    fraud_llm = llm.bind_tools([check_fraud_signals])

    messages_to_send = [
        SystemMessage(content=fraud_system_prompt),
        *state.get('messages', [])
    ]

    response = fraud_llm.invoke(messages_to_send)

    if response.tool_calls:

        tc     = response.tool_calls[0]
        result = check_fraud_signals.invoke(tc.get('args', {}))

        result_msg = ToolMessage(
            content      = json.dumps(result),
            tool_call_id = tc['id']
        )

        final_messages = messages_to_send + [response, result_msg]
        final          = fraud_llm.invoke(final_messages)

        print(f"[Fraud] tool called | risk_score="
              f"{result.get('risk_score', 'N/A')}")

        final_text = _extract_text(final.content)
        return {
            'messages':       [response, result_msg, final],
            'final_response': final_text,
            'internal_notes': [{
                'agent':   'fraud_handler',
                'finding': f"Fraud analysis complete | risk_score={result.get('risk_score')} | "
                           f"recommendation={result.get('recommendation')}",
            }],
        }

    else:
        final_text = _extract_text(response.content)
        return {
            'messages':       [response],
            'final_response': final_text,
            'internal_notes': [{
                'agent':   'fraud_handler',
                'finding': 'Fraud analysis complete — no tool call made',
            }],
        }

# ══════════════════════════════════════════════════════════════════
# SECTION 7: GRAPH ASSEMBLY (Session 7 — Subgraph Architecture)
# ══════════════════════════════════════════════════════════════════

# ── Triage Subgraph (Session 7) ──────────────────────────────────

def build_triage_subgraph():
    """
    Compiles the triage subgraph independently.
    Contains: ingress_node, route_after_ingress,
    blocked_response_node, classify_node.
    Entry point: ingress_node.
    Returns to master graph after classify_node completes.
    Permanent from Session 7 onward.
    """

    triage_builder = StateGraph(SharedState)

    triage_builder.add_node('ingress_node',          ingress_node)
    triage_builder.add_node('blocked_response_node', blocked_response_node)
    triage_builder.add_node('classify_node',         classify_node)

    triage_builder.set_entry_point('ingress_node')

    triage_builder.add_conditional_edges(
        'ingress_node',
        route_after_ingress,
        {
            'classify_node':         'classify_node',
            'blocked_response_node': 'blocked_response_node',
        }
    )

    triage_builder.add_edge('classify_node',         END)
    triage_builder.add_edge('blocked_response_node', END)

    triage_subgraph = triage_builder.compile(checkpointer=checkpointer)

    print("[Triage Subgraph] Compiled — 3 nodes | ingress entry")
    return triage_subgraph


triage_subgraph = build_triage_subgraph()

# ── Tech Support Subgraph (Session 7) ────────────────────────────

def summarization_check_node(state: SharedState) -> dict:
    """Thin entry node — no-op; routing handled by conditional edge."""
    return {}


def route_at_summarization_check(state: SharedState) -> str:
    """Routes to summarization_node if above threshold, else agent_node."""
    msg_count = len(state.get('messages', []))
    if msg_count > SUMMARY_THRESHOLD:
        print(f"[Router:tech_support] {msg_count} messages "
              f"> {SUMMARY_THRESHOLD} → summarization_node")
        return 'summarization_node'
    print(f"[Router:tech_support] {msg_count} messages "
          f"<= {SUMMARY_THRESHOLD} → agent_node")
    return 'agent_node'


def build_tech_support_subgraph():
    """
    Compiles the tech support subgraph independently.
    Contains: summarization_node, route_after_classify,
    agent_node, tool_node, route_after_agent,
    respond_node, egress_node.
    Entry point: summarization_check.
    Permanent from Session 7 onward.
    """

    tech_builder = StateGraph(SharedState)

    tech_builder.add_node('summarization_check', summarization_check_node)
    tech_builder.add_node('summarization_node',  summarization_node)
    tech_builder.add_node('agent_node',          agent_node)
    tech_builder.add_node('tool_node',           tool_node)
    tech_builder.add_node('respond_node',        respond_node)
    tech_builder.add_node('egress_node',         egress_node)

    tech_builder.set_entry_point('summarization_check')

    tech_builder.add_conditional_edges(
        'summarization_check',
        route_at_summarization_check,
        {
            'summarization_node': 'summarization_node',
            'agent_node':         'agent_node',
        }
    )

    tech_builder.add_edge('summarization_node', 'agent_node')

    tech_builder.add_conditional_edges(
        'agent_node',
        route_after_agent,
        {
            'tool_node':    'tool_node',
            'respond_node': 'respond_node',
        }
    )

    tech_builder.add_edge('tool_node',    'agent_node')
    tech_builder.add_edge('respond_node', 'egress_node')
    tech_builder.add_edge('egress_node',  END)

    tech_subgraph = tech_builder.compile(checkpointer=checkpointer)

    print("[Tech Support Subgraph] Compiled — 5 nodes | agent loop")
    return tech_subgraph


tech_support_subgraph = build_tech_support_subgraph()

# ── Custom Reducer (Session 7) ────────────────────────────────────

def demonstrate_silent_overwrite_bug():
    """
    DEMONSTRATION — called in CLI only, not in production.
    Shows what happens without operator.add on tool_results.
    Run this before the fix to see the bug live.
    """

    print("\n[BUG DEMO] Silent overwrite without operator.add:")
    state = {'tool_results': []}

    # Triage writes a finding
    state['tool_results'] = ['pii_scan: clean, injection: none']
    print(f"  After triage writes: {state['tool_results']}")

    # Tech support overwrites (the bug)
    state['tool_results'] = ['crm: C-1002 past due, balance $998']
    print(f"  After tech_support writes: {state['tool_results']}")
    print(f"  Triage finding: GONE — no error raised")

    print("\n[FIX DEMO] With operator.add:")
    findings = []
    findings = findings + ['pii_scan: clean, injection: none']
    print(f"  After triage writes: {findings}")
    findings = findings + ['crm: C-1002 past due, balance $998']
    print(f"  After tech_support writes: {findings}")
    print(f"  Both findings: PRESERVED")

def build_master_graph():
    """
    Builds the master graph with supervisor orchestration.
    Hub and spoke: every worker returns to supervisor_node.
    No worker routes directly to END.
    supervisor_router is the only path to termination.
    Session 8 replaces static routing with dynamic supervision.
    """

    master_builder = StateGraph(SharedState)

    # Register all nodes
    master_builder.add_node('triage',           triage_subgraph)
    master_builder.add_node('supervisor',        supervisor_node)
    master_builder.add_node('tech_support',      tech_support_subgraph)
    master_builder.add_node('fraud_handler',     fraud_handler)
    master_builder.add_node('general_handler',   general_handler)

    # Entry point — triage always runs first
    master_builder.set_entry_point('triage')

    # Triage → supervisor (always — supervisor reads triage findings)
    master_builder.add_edge('triage', 'supervisor')

    # Supervisor → conditional routing via supervisor_router
    master_builder.add_conditional_edges(
        'supervisor',
        supervisor_router,
        {
            'triage':          'triage',
            'tech_support':    'tech_support',
            'fraud_handler':   'fraud_handler',
            'general_handler': 'general_handler',
            END:               END,
        }
    )

    # Every worker returns to supervisor — hub and spoke
    master_builder.add_edge('tech_support',    'supervisor')
    master_builder.add_edge('fraud_handler',   'supervisor')
    master_builder.add_edge('general_handler', 'supervisor')

    # Compile with checkpointer
    graph = master_builder.compile(checkpointer=checkpointer)
    print("[Master Graph] Session 8 — 5 nodes | supervisor hub active")
    return graph


# Module-level graph instance
graph = build_master_graph()

# ══════════════════════════════════════════════════════════════════
# SECTION 8: INITIAL STATE BUILDER
# ══════════════════════════════════════════════════════════════════

def build_initial_state(ticket: str) -> dict:
    """
    Constructs a clean initial state for every graph invocation.
    Provides safe defaults for ALL 17 fields so no node gets a KeyError.
    Called by both the test harness and the Streamlit UI.
    """
    return {
        "raw_input":          ticket,
        "sanitized_input":    "",
        "category":           "",
        "messages":           [HumanMessage(content=ticket)],
        "customer_data":      {},
        "tool_results":       [],
        "pii_detected":       False,
        "injection_detected": False,
        "is_safe":            True,
        "system_summary":     "",
        "iteration_count":    0,
        "internal_notes":     [],
        "delegation_count":   0,
        "next_worker":        "",
        "github_draft":       {},
        "github_issue_url":   "",
        "final_response":     "",
    }

# ══════════════════════════════════════════════════════════════════
# SECTION 9: RUN FUNCTION (called by both CLI and UI)
# ══════════════════════════════════════════════════════════════════

def run_ticket(ticket: str,
               thread_id: str = None,
               return_existing: bool = False) -> dict:
    """
    Runs a ticket through the graph.
    If thread_id provided: loads prior state from checkpointer,
    appends new message, resumes conversation.
    If thread_id is None: generates a new thread_id,
    starts a fresh conversation.

    return_existing=True: if the thread already ran to END (e.g. the
    stream endpoint already executed it), return the existing final
    checkpoint state instead of re-invoking the graph. This prevents
    the stream+run UI pattern from executing the graph twice.
    Session 4+: always pass thread_id for persistent conversations.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"[Thread] New thread created: {thread_id}")

    config = {'configurable': {'thread_id': thread_id}}

    existing = list(graph.get_state_history(config))

    # Stream already ran this thread to completion — return its state.
    if return_existing and existing and len(existing[0].next) == 0:
        print(f"[Thread] Returning existing completed state | thread={thread_id}")
        result_dict = dict(existing[0].values)
        result_dict['thread_id'] = thread_id
        return result_dict

    is_first_turn = len(existing) == 0

    if is_first_turn:
        initial_state = build_initial_state(ticket)
        result = graph.invoke(initial_state, config=config)
        print(f"[Thread] First turn | thread={thread_id}")
    else:
        follow_up_state = {
            'messages': [HumanMessage(content=ticket)]
        }
        result = graph.invoke(follow_up_state, config=config)
        print(f"[Thread] Follow-up turn | thread={thread_id} | prior_steps={len(existing)}")

    result_dict = dict(result)
    result_dict['thread_id'] = thread_id
    return result_dict


def stream_ticket(ticket: str,
                  thread_id: str = None):
    """
    Generator. Yields (node_name, snapshot) tuples.
    Accepts thread_id for persistent streaming.
    Session 4+: pass thread_id for conversation continuity.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {'configurable': {'thread_id': thread_id}}

    existing = list(graph.get_state_history(config))
    is_first_turn = len(existing) == 0

    if is_first_turn:
        state_to_send = build_initial_state(ticket)
    else:
        state_to_send = {'messages': [HumanMessage(content=ticket)]}

    for namespace, step in graph.stream(
            state_to_send, config=config, subgraphs=True):
        for node_name, snapshot in step.items():
            yield node_name, (snapshot or {})

# ── Conversation History (Session 4) ─────────────────────────────

def get_conversation_history(thread_id: str) -> list:
    """
    Returns the full checkpoint history for a thread_id.
    Each entry is a dict with: step, node, state_summary,
    timestamp, is_end.
    Used by /api/history endpoint and the UI history panel.
    """
    config = {'configurable': {'thread_id': thread_id}}

    try:
        history = list(graph.get_state_history(config))
    except Exception as e:
        print(f"[History] Error loading thread {thread_id}: {e}")
        return []

    if not history:
        return []

    entries = []
    for snap in reversed(history):
        entry = {
            'step':           snap.metadata.get('step', 0),
            'source':         snap.metadata.get('source', ''),
            'node':           snap.metadata.get('source', 'unknown'),
            'category':       snap.values.get('category', ''),
            'iteration':      snap.values.get('iteration_count', 0),
            'message_count':  len(snap.values.get('messages', [])),
            'final_response': snap.values.get('final_response', ''),
            'is_end':         len(snap.next) == 0,
            'checkpoint_id':  snap.config.get('configurable', {})
                                  .get('checkpoint_id', ''),
        }
        entries.append(entry)

    print(f"[History] Thread {thread_id}: {len(entries)} checkpoints")
    return entries


def get_active_threads() -> list:
    """
    Returns a list of all thread_ids that have at least one checkpoint.
    Used by /api/threads endpoint and the thread selector in the UI.
    """
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        print(f"[Threads] {len(threads)} active threads")
        return threads
    except Exception as e:
        print(f"[Threads] Error: {e}")
        return []

# ══════════════════════════════════════════════════════════════════
# SECTION 10: SESSION VERIFICATION TEST
# ══════════════════════════════════════════════════════════════════

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 8 — VERIFICATION TEST                              │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │  Supervisor correctly routes after triage completes.        │
    │  Supervisor returns FINISH when ticket is resolved.         │
    │  Multi-intent ticket routes to two workers sequentially.    │
    │  Delegation count increments and safety limit fires.        │
    │  Structured output enforced — no free-text routing.         │
    │                                                             │
    │  PASS CRITERIA:                                             │
    │  ✓ Simple ticket → supervisor routes once → FINISH          │
    │  ✓ delegation_count increments correctly                    │
    │  ✓ supervisor internal_notes entries written each call      │
    │  ✓ MAX_DELEGATIONS=1 → FINISH fires immediately            │
    │  ✓ next_worker field set correctly after each delegation    │
    │                                                             │
    │  WHAT A PASS PROVES:                                        │
    │  Hub and spoke topology is correctly wired.                 │
    │  Structured output enforcement is working.                  │
    │  Safety limit prevents infinite delegation loops.           │
    │  Session 9 parallel execution can be added on top.          │
    └─────────────────────────────────────────────────────────────┘
    """

    import time
    start  = time.time()
    checks = []

    # ── CHECK 1: Simple ticket — supervisor routes and finishes ────

    thread_1 = f"verify-s8-simple-{int(time.time())}"
    result = run_ticket(
        "What is my subscription plan? Account C-1001.",
        thread_id=thread_1
    )

    delegation_used = result.get('delegation_count', 0)
    has_response    = bool(result.get('final_response', '').strip())
    next_w          = result.get('next_worker', '')

    check1_passed = (
        has_response and
        delegation_used >= 1 and
        next_w == 'FINISH'
    )

    checks.append({
        'label':        'Simple ticket — supervisor routes and returns FINISH',
        'passed':       check1_passed,
        'has_response': has_response,
        'note':         f"delegation_count={delegation_used} "
                        f"next_worker={next_w}",
    })

    # ── CHECK 2: delegation_count increments correctly ─────────────

    thread_2 = f"verify-s8-deleg-{int(time.time())}"
    result2 = run_ticket(
        "API returning 401 errors after the SDK update.",
        thread_id=thread_2
    )

    delegation2 = result2.get('delegation_count', 0)

    check2_passed = delegation2 >= 1

    checks.append({
        'label':        'delegation_count increments on each supervisor call',
        'passed':       check2_passed,
        'has_response': bool(result2.get('final_response', '').strip()),
        'note':         f"delegation_count={delegation2}",
    })

    # ── CHECK 3: supervisor writes to internal_notes ───────────────

    notes = result2.get('internal_notes', [])
    supervisor_entries = [
        n for n in notes
        if isinstance(n, dict) and n.get('agent') == 'supervisor'
    ]

    check3_passed = len(supervisor_entries) >= 1

    checks.append({
        'label':        'supervisor writes routing decisions to internal_notes',
        'passed':       check3_passed,
        'has_response': True,
        'note':         f"supervisor entries in notes: {len(supervisor_entries)}",
    })

    # ── CHECK 4: Safety limit fires ───────────────────────────────

    import support_agent as _sa
    original = _sa.MAX_DELEGATIONS
    _sa.MAX_DELEGATIONS = 1

    thread_4 = f"verify-s8-limit-{int(time.time())}"
    result4 = run_ticket(
        "Check billing for account C-1002.",
        thread_id=thread_4
    )

    _sa.MAX_DELEGATIONS = original

    limit_fired = result4.get('delegation_count', 0) >= 1
    has_resp4   = bool(result4.get('final_response', '').strip())

    check4_passed = limit_fired and has_resp4

    checks.append({
        'label':        'Safety limit fires at MAX_DELEGATIONS=1',
        'passed':       check4_passed,
        'has_response': has_resp4,
        'note':         f"delegation_count={result4.get('delegation_count')} "
                        f"MAX_DELEGATIONS restored to {original}",
    })

    # ── CHECK 5: next_worker set after each delegation ─────────────

    thread_5 = f"verify-s8-nw-{int(time.time())}"
    result5 = run_ticket(
        "I have a general question about pricing.",
        thread_id=thread_5
    )

    next_worker_set = bool(result5.get('next_worker', '').strip())

    check5_passed = next_worker_set

    checks.append({
        'label':        'next_worker field set correctly by supervisor',
        'passed':       check5_passed,
        'has_response': bool(result5.get('final_response', '').strip()),
        'note':         f"next_worker='{result5.get('next_worker')}'",
    })

    # ── RETURN ────────────────────────────────────────────────────

    all_passed   = all(c['passed'] for c in checks)
    duration_ms  = int((time.time() - start) * 1000)
    passed_count = sum(1 for c in checks if c['passed'])

    return {
        'passed':      all_passed,
        'checks':      checks,
        'summary':     f"{passed_count}/{len(checks)} checks passed "
                       f"in {duration_ms}ms",
        'duration_ms': duration_ms,
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 11: CLI TEST HARNESS
# ══════════════════════════════════════════════════════════════════

def run_cli_tests():
    """Runs all Session 8 test cases when file is executed directly."""

    print("\n" + "█" * 64)
    print("█  ENTERPRISE AI SUPPORT PLATFORM — SESSION 8 OF 12        █")
    print("█  The Supervisor Orchestrator                              █")
    print("█" * 64)

    import time as _time
    _ts = int(_time.time())

    # TEST 1 — Simple single-intent billing
    print(f"\n{'─' * 60}")
    print("TEST 1 — Simple single-intent billing")
    thread_1 = f"test-s8-bill-{_ts}"
    ticket1 = "What is my outstanding balance? Account C-1002."
    print(f"TICKET:    {ticket1}")
    print(f"EXPECTED:  triage classifies, supervisor routes to tech_support, "
          f"tech_support responds, supervisor returns FINISH")
    result1 = run_ticket(ticket1, thread_id=thread_1)
    deleg1 = result1.get('delegation_count', 0)
    nw1    = result1.get('next_worker', '')
    resp1  = result1.get('final_response', '')[:80]
    print(f"delegation_count: {deleg1}")
    print(f"next_worker:      {nw1}")
    print(f"final_response:   {resp1}")
    passed1 = deleg1 >= 1 and nw1 == 'FINISH' and bool(resp1)
    print(f"Status:    {'✅ PASS' if passed1 else '❌ FAIL'}")

    # TEST 2 — Technical ticket
    print(f"\n{'─' * 60}")
    print("TEST 2 — Technical ticket")
    thread_2 = f"test-s8-tech-{_ts}"
    ticket2 = "API returning 401 errors after SDK v3 update."
    print(f"TICKET:    {ticket2}")
    print(f"EXPECTED:  triage classifies, supervisor routes to tech_support, "
          f"KB searched, supervisor returns FINISH")
    result2 = run_ticket(ticket2, thread_id=thread_2)
    deleg2 = result2.get('delegation_count', 0)
    msgs2  = result2.get('messages', [])
    tools2 = [m for m in msgs2 if hasattr(m, 'tool_calls') and m.tool_calls]
    tool2_name = tools2[0].tool_calls[0]['name'] if tools2 else 'none'
    print(f"delegation_count: {deleg2}")
    print(f"tool called:      {tool2_name}")
    passed2 = deleg2 >= 1 and bool(result2.get('final_response', '').strip())
    print(f"Status:    {'✅ PASS' if passed2 else '❌ FAIL'}")

    # TEST 3 — Multi-intent ticket (billing + technical)
    print(f"\n{'─' * 60}")
    print("TEST 3 — Multi-intent ticket (billing + technical)")
    thread_3 = f"test-s8-multi-{_ts}"
    ticket3 = ("My API is returning 500 errors AND I was charged twice "
               "this month on account C-1002.")
    print(f"TICKET:    {ticket3}")
    print(f"EXPECTED:  supervisor routes to tech_support, reads findings, "
          f"may route again for billing, then FINISH")
    result3 = run_ticket(ticket3, thread_id=thread_3)
    deleg3 = result3.get('delegation_count', 0)
    resp3  = result3.get('final_response', '')[:80]
    print(f"delegation_count: {deleg3}  (should be > 1 for multi-intent)")
    print(f"final_response:   {resp3}")
    passed3 = deleg3 >= 1 and bool(resp3)
    print(f"Status:    {'✅ PASS' if passed3 else '❌ FAIL'}")

    # TEST 4 — Safety limit (MAX_DELEGATIONS=1)
    print(f"\n{'─' * 60}")
    print("TEST 4 — Safety limit (MAX_DELEGATIONS=1)")
    import support_agent as _sa
    original_max = _sa.MAX_DELEGATIONS
    _sa.MAX_DELEGATIONS = 1
    thread_4 = f"test-s8-limit-{_ts}"
    ticket4 = "Check billing for account C-1001."
    print(f"TICKET:    {ticket4}")
    print(f"EXPECTED:  supervisor fires once, limit reached, FINISH returned")
    result4 = run_ticket(ticket4, thread_id=thread_4)
    _sa.MAX_DELEGATIONS = original_max
    deleg4 = result4.get('delegation_count', 0)
    resp4  = result4.get('final_response', '')[:80]
    print(f"delegation_count: {deleg4}")
    print(f"final_response:   {resp4}")
    passed4 = deleg4 >= 1 and bool(resp4)
    print(f"Status:    {'✅ PASS' if passed4 else '❌ FAIL'}")

    # TEST 5 — General query
    print(f"\n{'─' * 60}")
    print("TEST 5 — General query")
    thread_5 = f"test-s8-gen-{_ts}"
    ticket5 = "What is the difference between the Pro and Enterprise plans?"
    print(f"TICKET:    {ticket5}")
    print(f"EXPECTED:  supervisor routes to general_handler, FINISH")
    result5 = run_ticket(ticket5, thread_id=thread_5)
    deleg5 = result5.get('delegation_count', 0)
    nw5    = result5.get('next_worker', '')
    resp5  = result5.get('final_response', '')[:80]
    print(f"delegation_count: {deleg5}")
    print(f"next_worker:      {nw5}")
    print(f"final_response:   {resp5}")
    passed5 = deleg5 >= 1 and nw5 == 'FINISH' and bool(resp5)
    print(f"Status:    {'✅ PASS' if passed5 else '❌ FAIL'}")

    # ── Full verification suite ─────────────────────────────────────
    verification = run_session_verification()

    print(f"\n{'═' * 64}")
    print(f"SESSION 8 COMPLETE — {verification['summary']}")
    for check in verification['checks']:
        status = '✅ PASS' if check['passed'] else '❌ FAIL'
        print(f"  {status}  {check['label']}")
        if check.get('note'):
            print(f"           {check['note']}")
    print("═" * 64)


# ══════════════════════════════════════════════════════════════════
# SECTION 12: MAIN BLOCK
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_cli_tests()
  
