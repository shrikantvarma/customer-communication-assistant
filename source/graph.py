"""The workflow for one Gmail thread: triage -> draft -> review (interrupt).

Only needs-reply threads go on to a draft; notify and ignore threads end after triage.

A plain `StateGraph`, not an agent. One LangGraph thread per Gmail thread, keyed by the Gmail
thread id. Nodes are plain functions over plain-dict state so the Postgres checkpointer can
store it; the AI nodes call `Model`, and Gmail is only read (`GmailAdapter` has no write
operation). The graph never writes to Gmail: the draft approver does, when the user approves a
request the app made (#19), and the app then records the written draft with `draft_approved`.

Review loops are graph edges, so each node execution calls `interrupt()` at most once.
Where each review decision goes:

- edit: review -> review (the preference suggestion follows in the background, #36)
- refine: review -> draft -> suggest_preference -> review
- answer (the draft's question): review -> draft -> review
- restore: review -> review
- approve_suggestion: review -> remember -> review; dismiss_suggestion: review -> review
- draft_approved, gmail_edited, gmail_draft_deleted: review -> review (the thread stays in
  review while a Gmail draft exists, #9)
- new_mail: review -> refresh_thread -> review
- dismiss: review -> END
- gmail_sent: review -> END (the user sent the reply from Gmail, #10)

Once a Gmail draft exists, a new version in the workspace does not change it: the user sends
the new version for approval, and the approver updates that same draft (#19). The app reads
the Gmail draft before each workspace action and resumes review with `gmail_edited` or
`gmail_draft_deleted` when Gmail changed under us (#9), or with `gmail_sent` when the draft is
gone because the user sent it (#10).

New mail on a thread in review starts no second run. Ingest tells the app, which resumes
review with `new_mail`; `refresh_thread` reads the whole thread from Gmail again and marks it
updated since the draft was written (#13). The reply is left as it is, and the next draft or
refine works from the full updated thread, which clears the mark. New mail from someone else
on a thread whose run ended (sent, notify, ignored, dismissed) runs it again from triage on the
same LangGraph thread, with `next_reply_cycle` as the input.

Only `remember` writes to the store (the memory profiles); the other nodes read it.

The preference suggestion after the user's own edit is no node run: leaving the thread with
changes (#45) and Send for approval go back to review at once, and the app asks for the
suggestion afterwards (`suggestion_after`), writing it into the run as if
`suggest_preference` had (#36).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, Self, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, model_validator

from langchain_email.gmail import EmailThread, GmailAdapter
from langchain_email.model import Model, Triage, answer_instruction
from langchain_email.profiles import (
    PROFILE_NAMES,
    read_owner,
    read_profile,
    save_profile,
    with_preference,
)

VersionAuthor = Literal["ai", "user"]


class Version(TypedDict):
    """One saved state of the reply text.

    A thread's versions are an append-only list in graph state (`ThreadState.versions`, with an
    `operator.add` reducer): nodes return new versions and no code path rewrites or removes
    one. List order is the order they were made, so `versions[-1]` is the current reply and a
    version's number is its index + 1. `author` says who made it: "ai" for the first draft and
    each refinement, "user" for the user's edits and restores. `instruction` is set only on a
    refinement the user asked for; `answer` (what they typed) and `picked` (the suggested answer
    they picked) only on one made from the user's reply to the draft's question. Undo adds a
    copy of the chosen earlier version, with `restored_from` set
    to that version's number, so later versions stay in the list. Gmail draft sync (#9) records
    Gmail-side edits as "user" versions with `edited_in_gmail` set.
    """

    author: VersionAuthor
    text: str
    instruction: NotRequired[str]
    answer: NotRequired[str]
    picked: NotRequired[str]
    restored_from: NotRequired[int]
    edited_in_gmail: NotRequired[bool]


class ThreadState(TypedDict):
    """Each node reads the keys earlier nodes wrote and returns only the keys it changes."""

    thread_id: str
    thread: dict[str, Any]  # EmailThread.model_dump(), written by triage and refresh_thread
    # New mail arrived on the thread after the current AI version was written (#13).
    thread_updated: NotRequired[bool]
    triage: NotRequired[dict[str, Any]]  # Triage.model_dump()
    versions: Annotated[list[Version], operator.add]  # append-only; needs-reply threads only
    # The latest ReviewDecision.model_dump(); None until the first one of the reply cycle.
    decision: NotRequired[dict[str, Any] | None]
    model: NotRequired[str | None]  # the model picked for this reply's refinements (#8)
    # The draft's one focused question for the user, until they answer it (None: nothing asked).
    question: NotRequired[str | None]
    # Up to three short answers to `question` the user can tap instead of typing (#44).
    suggested_answers: NotRequired[list[str]]
    gmail_draft_id: NotRequired[str | None]  # None once the draft was deleted in Gmail
    gmail_draft_text: NotRequired[str]  # the text the Gmail draft last had, as far as we know
    # When the Gmail draft was created (ISO 8601, UTC). Only Sent mail from after this can be
    # that draft sent (#10).
    gmail_draft_created_at: NotRequired[str | None]
    # The preference suggested after the user's latest correction (PreferenceSuggestion dump),
    # waiting for approve or dismiss; None once answered or when there was nothing to suggest.
    suggestion: NotRequired[dict[str, Any] | None]


Update = dict[str, Any]


def current_text(state: ThreadState | dict[str, Any]) -> str:
    """The reply as it stands: the newest version's text ("" before the first draft)."""
    versions = state.get("versions") or []
    return versions[-1]["text"] if versions else ""


class ReviewDecision(BaseModel):
    """The user's decision at review.

    - `edit`: save `text` as the user's new version and stay in review.
    - `refine`: have the model rework the current version following `instruction`. `model`,
      if set, is the model picked for this thread; it is kept and used for later refines too.
    - `restore`: make version number `version` current again (as a new copy) and stay in review.
      The caller checks the version exists; review only knows the thread's versions on resume.
    - `draft_approved`: the draft approver wrote `text` to the Gmail draft `draft_id` at `at`
      (ISO 8601), because the user approved it (#19). Only the app sends this, from the
      approval's outcome; the graph itself never writes to Gmail.
    - `answer`: reply to the draft's question with `picked` (a suggested answer), `answer` (what
      the user typed), or both; the model works out what answers the question and what asks
      for a change, and reworks the current version.
    - `dismiss`: end the run with no Gmail draft; the thread leaves the review list. A notify
      thread is dismissed by writing this decision into its finished run.
    - `approve_suggestion`: save the pending suggested preference to its profile.
    - `dismiss_suggestion`: drop it; both profiles stay as they are.
    - `gmail_edited`: the user changed the Gmail draft to `text`; record it as their newest version.
    - `gmail_draft_deleted`: the Gmail draft is gone; the reply is only prepared again.
    - `gmail_sent`: the Gmail draft is gone because the user sent the reply from Gmail; the
      run ends. Only the app's sent check (#10) decides this, from the user's Sent mail.
    - `new_mail`: new mail arrived on the thread; read it again from Gmail. Ingest sends this,
      through the app, not the user.

    After a Gmail draft exists, edit, refine, answer and restore leave it as it is until the
    user sends the new version for approval.
    """

    action: Literal[
        "draft_approved",
        "edit",
        "refine",
        "restore",
        "answer",
        "dismiss",
        "approve_suggestion",
        "dismiss_suggestion",
        "gmail_edited",
        "gmail_draft_deleted",
        "gmail_sent",
        "new_mail",
    ]
    text: str | None = None
    instruction: str | None = None
    model: str | None = None
    version: int | None = None
    answer: str | None = None
    picked: str | None = None
    draft_id: str | None = None
    at: str | None = None

    @model_validator(mode="after")
    def _fields_match_action(self) -> Self:
        if self.action == "edit" and not (self.text and self.text.strip()):
            raise ValueError("an edit needs the new text")
        if self.action == "refine" and not (self.instruction and self.instruction.strip()):
            raise ValueError("a refine needs an instruction")
        if self.action == "restore" and not (self.version and self.version >= 1):
            raise ValueError("a restore needs a version number")
        said = (self.answer or "").strip() or (self.picked or "").strip()
        if self.action == "answer" and not said:
            raise ValueError("an answer can't be empty")
        if self.action == "gmail_edited" and self.text is None:
            raise ValueError("a Gmail edit needs the draft's text")
        if self.action == "draft_approved" and not (self.draft_id and self.text is not None):
            raise ValueError("an approved draft needs its id and text")
        return self


NOTE_TO_SELF = Triage(classification="ignore", reason="You sent this only to yourself.")
"""The triage of a thread the user sent only to their own address, made without the model."""

USER_NEEDS_REPLY = Triage(
    classification="needs_reply", reason="You marked this as needing a reply."
)
"""The triage a user's "needs a reply" override writes. Applied as if the triage node wrote it
(`update_state(..., as_node="triage")`), so the run continues along the triage edge to draft."""


def next_reply_cycle(thread_id: str) -> Update:
    """The input that runs a thread whose run ended through triage, draft and review again.

    Someone wrote on a thread that was sent, notify, ignored or dismissed: it is new mail that
    may need a reply. The run starts over at triage on the same LangGraph thread, so the
    versions so far stay in the history (they are append-only). Everything that belonged to the
    last reply is reset: its decision, question, suggestion, Gmail draft (a sent or deleted one
    is gone from Gmail) and the model picked for it, so the new mail goes to the default model
    until the user picks another.
    """
    return {
        "thread_id": thread_id,
        "decision": None,
        "question": None,
        "suggested_answers": [],
        "suggestion": None,
        "model": None,
        "thread_updated": False,
        "gmail_draft_id": None,
        "gmail_draft_text": "",
        "gmail_draft_created_at": None,
    }


def decision_of(state: ThreadState | dict[str, Any]) -> ReviewDecision | None:
    """The thread's latest review decision, validated; None before the first of this reply.

    State keeps the decision as a plain dict (the checkpointer stores plain data); read it
    through here rather than by its keys.
    """
    raw = state.get("decision")
    return ReviewDecision.model_validate(raw) if raw else None


def suggestion_after(
    model: Model,
    state: ThreadState | dict[str, Any],
    profiles: dict[str, str],
    *,
    default_model: str,
) -> dict[str, Any] | None:
    """The preference to suggest after the user's latest correction, the newest two versions
    (a PreferenceSuggestion dump), or None. A refine's instruction goes with it. It only reads:
    saving happens in `remember`, after the user approves."""
    versions = state.get("versions") or []
    if len(versions) < 2:
        return None
    decision = decision_of(state)
    suggestion = model.suggest_preference(
        EmailThread.model_validate(state["thread"]),
        before=versions[-2]["text"],
        after=versions[-1]["text"],
        instruction=decision.instruction if decision and decision.action == "refine" else None,
        profiles=profiles,
        model=state.get("model") or default_model,  # the thread's picked model (#8)
    )
    return suggestion.model_dump() if suggestion else None


def config_for(thread_id: str) -> dict[str, Any]:
    """The LangGraph run config for a Gmail thread."""
    return {"configurable": {"thread_id": thread_id}}


def build_graph(*, model: Model, gmail: GmailAdapter, default_model: str) -> StateGraph:
    """The uncompiled graph. Compile it with a checkpointer (and store) to run it."""

    def triage(state: ThreadState, *, store: BaseStore) -> Update:
        thread = gmail.get_thread(state["thread_id"])
        if thread.only_to_myself:  # a note to self: nothing to reply to, no model needed
            result = NOTE_TO_SELF
        else:
            profile = read_profile(store, "triage")
            result = model.triage(thread, model=default_model, profile=profile)
        return {"thread": thread.model_dump(), "triage": result.model_dump()}

    def after_triage(state: ThreadState) -> str:
        needs_reply = state.get("triage", {}).get("classification") == "needs_reply"
        return "draft" if needs_reply else END

    def draft(state: ThreadState, *, store: BaseStore) -> Update:
        thread = EmailThread.model_validate(state["thread"])
        profile = read_profile(store, "reply_style")
        owner = read_owner(store)
        decision = decision_of(state)
        if decision is not None and decision.action in ("refine", "answer"):
            # Both rework the user's latest version; an answer (the question, the pick and what
            # was typed, together) is turned into the instruction.
            if decision.action == "refine" and decision.instruction:
                instruction = decision.instruction
                version = Version(author="ai", text="", instruction=instruction)
                update: Update = {}
            else:
                typed, picked = (decision.answer or "").strip(), (decision.picked or "").strip()
                instruction = answer_instruction(state.get("question"), picked, typed)
                version = Version(author="ai", text="")
                if typed:
                    version["answer"] = typed
                if picked:
                    version["picked"] = picked
                update = {"question": None, "suggested_answers": []}
            chosen = state.get("model") or default_model
            version["text"] = model.refine(
                thread,
                current_text(state),
                instruction,
                model=chosen,
                profile=profile,
                owner=owner,
            )
            return {**update, "versions": [version], "thread_updated": False}
        drafted = model.draft_reply(thread, model=default_model, profile=profile, owner=owner)
        return {
            "versions": [Version(author="ai", text=drafted.reply)],
            "question": drafted.question,
            "suggested_answers": drafted.suggested_answers,
            "thread_updated": False,
        }

    def after_draft(state: ThreadState) -> str:
        decision = decision_of(state)
        refined = decision is not None and decision.action == "refine"
        return "suggest_preference" if refined else "review"

    def suggest_preference(state: ThreadState, *, store: BaseStore) -> Update:
        # Runs after a refine, which waits on the model anyway. It only reads the store.
        profiles = {name: read_profile(store, name) for name in PROFILE_NAMES}
        return {"suggestion": suggestion_after(model, state, profiles, default_model=default_model)}

    def remember(state: ThreadState, *, store: BaseStore) -> Update:
        # The user approved the suggestion. `with_preference` is idempotent, so a rerun is safe.
        suggestion = state.get("suggestion")
        if suggestion:
            name = suggestion["profile"]
            save_profile(
                store, name, with_preference(read_profile(store, name), suggestion["preference"])
            )
        return {"suggestion": None}

    def review(state: ThreadState) -> Update:
        # Everything before interrupt() reruns on resume, so nothing here has side effects.
        # interrupt(response_schema=...) does not validate the resume value; we do.
        payload = {"draft": current_text(state), "question": state.get("question")}
        raw = interrupt(payload, response_schema=ReviewDecision)
        decision = ReviewDecision.model_validate(raw)
        update: Update = {"decision": decision.model_dump()}
        if decision.action == "edit" and decision.text is not None:
            update["versions"] = [Version(author="user", text=decision.text)]
            # A suggestion made for the text before this edit no longer explains the reply;
            # the one for this edit comes later, from the background (#36).
            update["suggestion"] = None
        if decision.action == "refine" and decision.model:
            update["model"] = decision.model
        if decision.action == "restore" and decision.version is not None:
            earlier = state["versions"][decision.version - 1]
            update["versions"] = [
                Version(author="user", text=earlier["text"], restored_from=decision.version)
            ]
        if decision.action == "dismiss_suggestion":
            update["suggestion"] = None
        if decision.action == "gmail_edited" and decision.text is not None:
            update["versions"] = [Version(author="user", text=decision.text, edited_in_gmail=True)]
            update["gmail_draft_text"] = decision.text
        if decision.action == "gmail_draft_deleted":
            update["gmail_draft_id"] = None
        if decision.action == "draft_approved" and decision.text is not None:
            if decision.draft_id != state.get("gmail_draft_id"):
                # A new draft; an update keeps the time the draft was first created (#10).
                update["gmail_draft_created_at"] = decision.at
            update["gmail_draft_id"] = decision.draft_id
            update["gmail_draft_text"] = decision.text
        return update

    def after_review(state: ThreadState) -> str:
        decision = decision_of(state)
        if decision is None:
            raise RuntimeError("review writes its decision before this edge runs")
        match decision.action:
            case "refine" | "answer":
                return "draft"
            case "approve_suggestion":
                return "remember"
            case "new_mail":
                return "refresh_thread"
            case "dismiss" | "gmail_sent":
                return END
            case (
                "edit"
                | "restore"
                | "dismiss_suggestion"
                | "draft_approved"
                | "gmail_edited"
                | "gmail_draft_deleted"
            ):
                return "review"

    def refresh_thread(state: ThreadState) -> Update:
        # Plain code: read the whole thread again, as triage did. Reading is safe to rerun.
        thread = gmail.get_thread(state["thread_id"])
        before = EmailThread.model_validate(state["thread"])
        grew = len(thread.messages) > len(before.messages)
        return {
            "thread": thread.model_dump(),
            "thread_updated": grew or state.get("thread_updated", False),
        }

    graph = StateGraph(ThreadState)
    graph.add_node("triage", triage)
    graph.add_node("draft", draft)
    graph.add_node("review", review)
    graph.add_node("suggest_preference", suggest_preference)
    graph.add_node("remember", remember)
    graph.add_node("refresh_thread", refresh_thread)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", after_triage, ["draft", END])
    graph.add_conditional_edges("draft", after_draft, ["suggest_preference", "review"])
    graph.add_conditional_edges(
        "review",
        after_review,
        {
            "review": "review",
            "draft": "draft",
            "remember": "remember",
            "refresh_thread": "refresh_thread",
            END: END,
        },
    )
    graph.add_edge("suggest_preference", "review")
    graph.add_edge("remember", "review")
    graph.add_edge("refresh_thread", "review")
    return graph
