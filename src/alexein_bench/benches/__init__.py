"""Every benchmark this harness runs, by the name `alexein-bench run` takes."""

from __future__ import annotations

from alexein_bench.benches.beam import Beam
from alexein_bench.benches.locomo import Locomo
from alexein_bench.benches.locomo_plus import LocomoPlus
from alexein_bench.benches.longmemeval import LongMemEval
from alexein_bench.benches.mab import Mab
from alexein_bench.benches.rhelm import Rhelm
from alexein_bench.core import Bench

BENCHES: dict[str, Bench] = {
    b.name: b for b in (Rhelm(), Mab(), Locomo(), LocomoPlus(), LongMemEval(), Beam())
}
