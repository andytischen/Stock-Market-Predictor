"""Named bundles of hypothetical moves for ``predict --scenario``.

A scenario is nothing more than a set of simple returns keyed by instrument,
the same input ``--shock`` takes one instrument at a time. It exists so the
recurring macro events that move several instruments at once can be replayed
without spelling out each leg, and so the sizes used are written down next to
the reasoning for them rather than living in a shell history.

Sizes are one-session moves, deliberately modest: they describe the market's
reaction to the announcement, not the multi-week drift that follows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """A hypothetical macro event expressed as simple returns per instrument."""

    name: str
    description: str
    moves: dict[str, float] = field(default_factory=dict)

    def shocks(self) -> dict[str, float]:
        """The moves as log returns, the form ``shocked_row`` expects."""
        return {symbol: float(np.log1p(move)) for symbol, move in self.moves.items()}


# A supply increase is a bearish crude shock: more barrels against unchanged
# demand. Brent moves slightly more than WTI because the incremental barrels
# are seaborne and priced off it. The dollar firms a little on the cheaper
# energy import bill, and the disinflationary read pulls the long end of the
# curve down with it. Successive increases land softer than the first — by the
# sixth the market has largely priced the cadence — hence the modest sizes.
_OPEC_SUPPLY_INCREASE = Scenario(
    name="opec-supply-increase",
    description="OPEC+ raises output quotas: crude sells off, dollar firms, yields ease",
    moves={"CL=F": -0.035, "BZ=F": -0.04, "DX-Y.NYB": 0.002, "^TNX": -0.01},
)

# The mirror image: a production cut tightens supply and lifts crude, with the
# inflation read pushing yields the other way.
_OPEC_SUPPLY_CUT = Scenario(
    name="opec-supply-cut",
    description="OPEC+ cuts output quotas: crude rallies, yields rise",
    moves={"CL=F": 0.05, "BZ=F": 0.055, "DX-Y.NYB": -0.002, "^TNX": 0.015},
)

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (_OPEC_SUPPLY_INCREASE, _OPEC_SUPPLY_CUT)}


def scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(SCENARIOS)
        raise KeyError(f"unknown scenario {name!r}; known scenarios: {known}") from exc
