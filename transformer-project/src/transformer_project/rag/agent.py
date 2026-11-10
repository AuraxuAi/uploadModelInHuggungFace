"""
Agentic RAG orchestrator: wires the DocumentStore (retrieval) together with
repeated calls to your own model, playing different "roles" per call
(planner / worker / aggregator). There is no special model capability
required for "agent" behavior — it's this orchestration code calling the
same model multiple times with different prompts, using the model's own
output at each step to decide what to do next.

IMPORTANT — read before expecting reliable results:
Reliability of the planning/decomposition step depends heavily on model
scale and dedicated instruction-tuning. At ~150-200M params with limited
SFT, expect the "planner" step in particular to be unreliable (the same
class of limitation you already observed in plain function completion —
see the calculate_mean example where logic didn't connect end-to-end).
This module is built to be swapped onto a larger, better-instruction-tuned
model (e.g. your planned 7B run) without changing its structure — only the
underlying `generate_fn` call needs to point at a different checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from transformer_project.rag.retrieval import Chunk, DocumentStore


@dataclass
class AgentConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    strategy: str = "sample"
    top_k_retrieval: int = 3


class RAGAgent:
    """
    generate_fn: a callable (prompt: str, **gen_kwargs) -> str, matching your
    project's transformer_project.inference.generate.generate() signature
    minus model/tokenizer/device (bind those with functools.partial before
    passing in — keeps this module decoupled from any specific checkpoint).
    """

    def __init__(self, generate_fn: Callable[..., str], store: DocumentStore, config: AgentConfig | None = None):
        self.generate_fn = generate_fn
        self.store = store
        self.config = config or AgentConfig()

    def _call_model(self, prompt: str) -> str:
        return self.generate_fn(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            strategy=self.config.strategy,
        )

    def answer_with_context(self, query: str) -> dict:
        """Single-step RAG: retrieve relevant chunks, inject as context, ask
        the model to answer. This is the reliable baseline — start here
        before attempting multi-step planning."""
        chunks = self.store.retrieve(query, top_k=self.config.top_k_retrieval)
        context = "\n\n".join(c.text for c in chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = self._call_model(prompt)
        return {"query": query, "context_used": [c.source for c in chunks], "response": response}

    def run_multi_step_task(self, task: str, max_subtasks: int = 3) -> dict:
        """Agentic multi-step: ask the model to break the task into
        subtasks, retrieve context and answer each one, then combine.

        Given the reliability caveat above, this is best treated as an
        experimental path to validate once you have a larger, better
        instruction-tuned model — the plan-parsing here is intentionally
        simple (newline-split, numbered-list-aware) rather than requiring
        strict structured output the current model likely can't reliably
        produce.
        """
        plan_prompt = (
            f"Break the following task into at most {max_subtasks} short, "
            f"numbered steps.\n\nTask: {task}\n\nSteps:"
        )
        plan_text = self._call_model(plan_prompt)
        subtasks = self._parse_subtasks(plan_text, max_subtasks)

        if not subtasks:
            # Planning failed to produce anything parseable — fall back to
            # treating the whole task as one retrieval+answer step, rather
            # than silently returning nothing.
            return {
                "task": task, "plan_raw": plan_text, "subtasks": [],
                "fallback": True, **self.answer_with_context(task),
            }

        step_results = [self.answer_with_context(sub) for sub in subtasks]

        combined_context = "\n\n".join(
            f"Step {i+1} ({r['query']}): {r['response']}" for i, r in enumerate(step_results)
        )
        final_prompt = (
            f"Original task: {task}\n\nStep results:\n{combined_context}\n\nFinal answer:"
        )
        final_answer = self._call_model(final_prompt)

        return {
            "task": task, "plan_raw": plan_text, "subtasks": subtasks,
            "step_results": step_results, "final_answer": final_answer, "fallback": False,
        }

    @staticmethod
    def _parse_subtasks(plan_text: str, max_subtasks: int) -> list[str]:
        import re
        lines = [l.strip() for l in plan_text.splitlines() if l.strip()]
        subtasks = []
        for line in lines:
            # Require an actual list marker (digit+separator, or -/*) —
            # a plain sentence with no marker is NOT a parsed subtask, it's
            # unstructured rambling that should trigger the fallback path
            # instead of being silently treated as "one subtask."
            match = re.match(r"^(?:\d+[.)]\s*|[-*]\s+)(.+)", line)
            if not match:
                continue
            cleaned = match.group(1).strip()
            if cleaned and len(cleaned) > 3:
                subtasks.append(cleaned)
            if len(subtasks) >= max_subtasks:
                break
        return subtasks
