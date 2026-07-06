"""Hard spend cap (ADR 0002): budgets table + enforcement middleware.

Fail-closed: no configured budget means no spend. Over-budget requests get a
structured 429 refusal — a tested product behaviour, not an error path.
"""
