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
- Pure GA, NPO, and SimNPO are separate named methods.
- GradDiff and `_grad_diff` variants add retain NLL.
- `_kl` variants add target-model KL regularization.
- `gamma` and `alpha` independently scale forget and retain terms.
- Unlearning uses the same QA-answer and statement examples used during
  acquisition, rather than targeting only one view of each acquired fact.
- Retain regularization data is disjoint from retain evaluation data.
- Fine-grained per-example metrics are always saved alongside aggregates.
