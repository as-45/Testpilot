"""
sim/device.py - the modem under test.

A Device is one software build of the modem: its build number plus the set
of faults that are active in it.  For a given test, the device works out
which active faults actually reach that test (right component, right
modulation) and turns them into:
  * simulator settings ("impairments" from sim/channel.py), and
  * a timing slowdown (the scheduler bug).

A healthy ("golden") build is simply Device(build_id=0).
"""

from dataclasses import dataclass, field

from sim.faults import COMPONENTS, FAULTS


@dataclass(frozen=True)
class Device:
    build_id: int
    active_faults: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        # accept any iterable, store as a frozenset, and reject typos early
        faults = frozenset(self.active_faults)
        unknown = faults - set(FAULTS)
        if unknown:
            raise ValueError(f"unknown fault(s): {sorted(unknown)}")
        object.__setattr__(self, "active_faults", faults)

    def faults_touching(self, test_components: tuple, modulation: str) -> list[str]:
        """
        Active faults that can affect this test: the fault's component is one
        the test exercises, and the fault applies to the test's modulation.
        """
        hits = []
        for name in sorted(self.active_faults):
            fault = FAULTS[name]
            if fault["component"] not in test_components:
                continue
            if fault["modulations"] is not None and modulation not in fault["modulations"]:
                continue
            hits.append(name)
        return hits

    def impairments(self, test_components: tuple, modulation: str) -> dict:
        """Simulator settings from every fault that reaches this test."""
        merged = {}
        for name in self.faults_touching(test_components, modulation):
            merged.update(FAULTS[name]["impairments"])
        return merged

    def slowdown(self, test_components: tuple, modulation: str) -> float:
        """How much slower this test runs on this build (scheduler bug)."""
        factor = 1.0
        for name in self.faults_touching(test_components, modulation):
            factor *= FAULTS[name]["slowdown"]
        return factor

    @property
    def is_golden(self) -> bool:
        return not self.active_faults


GOLDEN = Device(build_id=0)

__all__ = ["Device", "GOLDEN", "COMPONENTS"]