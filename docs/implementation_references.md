# Implementation References

Primary references:

- MUSE paper and `jaechan-repo/muse_bench` at commit
  `6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e` for benchmark criteria and metric
  definitions.
- `locuslab/open-unlearning` at commit
  `4ad738aaf60f6a4385f6e2506d01da99e76c31f3` for tested NPO, SimNPO,
  retain-loss, evaluator, and supplementary metric implementations.
- NPO and SimNPO papers in `docs/` for objective definitions and experimental
  cautions.

Important implementation decisions:

- NPO uses the frozen merged target model as its reference.
- SimNPO uses a length-normalized, reference-free forget objective.
- Retain NLL and retain KL are independent composable regularizers.
- Retain regularization data is disjoint from retain evaluation data.
- Fine-grained per-example metrics are always saved alongside aggregates.
