"""
sim/faults.py - the catalog of bugs we can inject into the simulated modem.

Each fault lives in ONE component and changes the simulator in ONE way.
Because we inject them ourselves, we always know the true cause of a
failure - that "ground truth" is what lets us score the ML model and the
triage agent honestly later.

Faults also MAY leave a hint line in the test log (not always - real logs
are messy), and harmless "noise" warnings are mixed in so that triage is a
real problem, not a simple text lookup.
"""

COMPONENTS = ("rf_frontend", "mapper", "sync", "equalizer", "scheduler")

# name -> what it is, where it lives, how it changes the simulator, which
# modulations it affects (None = all), and the hint it may leave in logs
FAULTS = {
    "rf_gain_mismatch": {
        "component": "rf_frontend",
        "description": "RF gain calibrated wrong: real signal 2 dB weaker than configured",
        "impairments": {"snr_offset_db": -2.0},
        "slowdown": 1.0,
        "modulations": None,      # None = affects every modulation
        "hint": "[WARN ]  rf: AGC gain mismatch 2.0 dB",
    },
    "mapper_scale": {
        "component": "mapper",
        "description": "16-QAM mapping table scaled 20% too small (QPSK path unaffected)",
        "impairments": {"tx_scale": 0.8},
        "slowdown": 1.0,
        "modulations": ("16QAM",),
        "hint": "[WARN ]  mapper: constellation energy 0.64 (expected 1.00)",
    },
    "sync_phase": {
        "component": "sync",
        "description": "Phase tracking broken: every symbol rotated by 12 degrees",
        "impairments": {"phase_offset_deg": 12.0},
        "slowdown": 1.0,
        "modulations": None,
        "hint": "[WARN ]  sync: residual phase error 12.1 deg",
    },
    "eq_disabled": {
        "component": "equalizer",
        "description": "Equaliser skipped: fading is never corrected",
        "impairments": {"equalizer_enabled": False},
        "slowdown": 1.0,
        "modulations": None,
        "hint": "[WARN ]  eq: channel estimate stale, equalisation bypassed",
    },
    "sched_slow": {
        "component": "scheduler",
        "description": "Scheduler regression: long runs take 2.5x longer and hit the watchdog",
        "impairments": {},
        "slowdown": 2.5,          # only applied to tests that exercise the scheduler
        "modulations": None,
        "hint": "[WARN ]  sched: task queue depth 48 (limit 16)",
    },
}

# Harmless warnings that appear in healthy logs too.  Some look like the real
# hints on purpose (same subsystem prefix) - a good triage must not be fooled.
NOISE_WARNINGS = (
    "[WARN ]  thermal: board temperature 61 C (limit 85 C)",
    "[WARN ]  log: buffer 80% full",
    "[WARN ]  rf: AGC settling took 2 frames",
    "[WARN ]  eq: channel estimate refreshed 1 frame late",
    "[WARN ]  sync: phase tracker re-locked once",
    "[WARN ]  sched: task queue depth 12 (limit 16)",
)

FLAKY_LINE = "[ERROR]  harness: lost connection to instrument, run aborted"


def fault_component(name: str) -> str:
    """Which component a fault lives in (raises a clear error for typos)."""
    if name not in FAULTS:
        raise KeyError(f"unknown fault {name!r}; known faults: {sorted(FAULTS)}")
    return FAULTS[name]["component"]


def faults_in_component(component: str) -> list[str]:
    """All fault names that live in a component (used by build history)."""
    if component not in COMPONENTS:
        raise KeyError(f"unknown component {component!r}; use one of {COMPONENTS}")
    return [n for n, f in FAULTS.items() if f["component"] == component]