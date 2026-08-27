"""FS-HSAC: feasible-support hybrid SAC on the Sysplorer FMU twin.

Paper §4 map (object → support → executable loop):

- §4.1 twin-closed loop: ``collector.FSHSACCollector.step_with_givesafe``
- §4.2–4.3 support + actor: ``action_support``, ``actor.FSHSACActor.act``
  (inference samples the chosen mode only)
- §4.4 critic + exact mode sum: ``critic``, ``algorithm.FSHSAC._exact_soft_value``
- §4.5 residual C_ψ: ``feasibility`` (penalty, not a second hard gate)
"""

from training.fs_hsac.algorithm import FSHSAC, ALGORITHM_VERSION
from training.fs_hsac.actor import FSHSACActor
from training.fs_hsac.critic import FSHSACCritic

__all__ = ["FSHSAC", "FSHSACActor", "FSHSACCritic", "ALGORITHM_VERSION"]
