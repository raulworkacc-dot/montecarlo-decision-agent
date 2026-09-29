"""Decision intelligence case study: ML counterfactuals + Monte Carlo + a Claude agent.

Pipeline: synthetic history -> conversion/value models -> counterfactual uplift per
business lever -> Monte Carlo over execution, demand and model uncertainty -> ranked
decisions -> decision memo (Claude tool-use agent, or a deterministic rule-based memo).
"""

__version__ = "1.0.0"
