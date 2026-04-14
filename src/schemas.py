"""Typed output schemas for each deliberation role (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProposeOutput(BaseModel):
    """Output from a Propose-role agent."""
    approach_description: str = Field(..., description="Brief name of the solution approach used (< 20 words)")
    reasoning: str = Field(..., description="Step-by-step reasoning using the assigned approach")
    solution: str = Field(..., description="Final answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the solution (0-1)")


class CritiqueOutput(BaseModel):
    """Output from a Critique-role agent."""
    target_proposal: str = Field(..., description="Which proposal is being critiqued")
    flaw_type: str = Field(..., description="Type of flaw: logical | computational | incomplete | wrong_approach")
    flaw_description: str = Field(..., description="Description of the flaw found")
    severity: str = Field(..., description="Severity: minor | major | fatal")
    does_not_propose_alternative: bool = Field(True, description="Must be true — critic does not propose solutions")


class VerifyOutput(BaseModel):
    """Output from a Verify-role agent."""
    claim_being_verified: str = Field(..., description="The specific claim being checked")
    verification_method: str = Field(..., description="Method used: symbolic | numerical | logical | example")
    result: str = Field(..., description="Result: pass | fail")
    specific_failure: str | None = Field(None, description="Details of failure if result is fail")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in verification (0-1)")


class InsightOutput(BaseModel):
    """Output from an Insight-role agent (information-shielded)."""
    new_approach: str = Field(..., description="A qualitatively different approach to the problem")
    why_different: str = Field(..., description="How this differs from likely existing proposals")
    reasoning: str = Field(..., description="Step-by-step reasoning using the new approach")
    solution: str = Field(..., description="Final answer from this approach")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the solution (0-1)")


class SynthesizeOutput(BaseModel):
    """Output from a Synthesize-role agent."""
    proposals_considered: list[str] = Field(..., description="Brief summary of each proposal considered")
    synthesis_method: str = Field(..., description="Method: best_of | combination | refinement")
    final_solution: str = Field(..., description="The synthesized final answer")
    justification: str = Field(..., description="Why this answer was chosen")


# Registry
ROLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "propose": ProposeOutput,
    "critique": CritiqueOutput,
    "verify": VerifyOutput,
    "insight": InsightOutput,
    "synthesize": SynthesizeOutput,
}
