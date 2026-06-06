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
- `method_overrides` preserves method-specific published hyperparameters instead
  of forcing GA, NPO, and SimNPO to share one setting.

## Hyperparameter Sources

- Original MUSE baseline scripts use `10` epochs, learning rate `1e-5`, and
  per-device batch size `4` for GA, NPO, and their regularized variants:
  <https://github.com/jaechan-repo/muse_bench/blob/main/baselines/scripts/unlearn_news.sh>
- OpenUnlearning's filled MUSE unlearning example uses NPO with learning rate
  `3e-5`, `10` epochs, `beta=0.1`, `gamma=1.0`, `alpha=1.0`, batch size `4`,
  and gradient accumulation `8`:
  <https://github.com/locuslab/open-unlearning/blob/main/configs/experiment/examples/muse_unlearn.yaml>
- The official SimNPO MUSE commands use `10` epochs, learning rate `1e-5`,
  forget coefficient `1.0`, retain coefficient `0.1`, and `beta=0.7` for News
  or `0.75` for Books:
  <https://github.com/OPTML-Group/Unlearn-Simple/tree/main/MUSE>

The experiment uses the News value `beta=0.7` because its generated facts are
short, heterogeneous factual records rather than long-form book passages.
This is an explicit transfer assumption, not a published optimum for Qwen or
LoRA.
- Unlearning uses the same QA-answer and statement examples used during
  acquisition, rather than targeting only one view of each acquired fact.
- Retain regularization data is disjoint from retain evaluation data.
- Fine-grained per-example metrics are always saved alongside aggregates.
