# Noise-floor calibration: investigation notes (2026-08)

Record of the noise-floor investigation, the code changes made, and the empirical
finding that settles which null to use. Source file: `scripts/behavioral_fingerprint/sequential_cp_comparison.py`.
A/B harness: `scripts/behavioral_fingerprint/noise_null_ab.py`.

## TL;DR

- The production noise floor is a **pooled split-half permutation test** with the
  **visitation-weighted** `strategic_shift`. Verified across three envs, this is the
  **correct choice**: family-wise FPR on Random policies is 5.6% (Taxi) and 3.6%
  (FrozenLake). MountainCar is the lone over-firer (~20%), but at magnitudes ~500x
  below any real signal.
- **Two alternatives were tested and rejected:** the target-checkpoint-only null
  (0% FPR on Taxi and FrozenLake, i.e. zero sensitivity) and removing the weighting
  (never systematically helps).
- **No rollback needed:** production already uses pooled/weighted. The changes that
  stand are the estimator fix, the M bump, the decomposition-floor pruning, and the
  speedups (below).

## Changes made to `sequential_cp_comparison.py` (all verified)

1. **Quantile estimator `method="higher"`** on the three per-metric floors and
   `zmax_p90/95/99`. This makes the threshold the permutation-exact (conformal)
   cutoff: with M null draws + 1 fresh observation all exchangeable, the level-a
   cutoff is the `ceil((M+1)(1-a))`-th order statistic, which `np.quantile(...,
   "higher")` reproduces at every M. numpy's default `"linear"` interpolation lands
   ~2 ranks lower and realizes ~9.1% FPR at M=20 instead of ~4.8% (simulation-proven,
   distribution-free across normal/expo/lognormal). `net_lo` uses `method="lower"`.

2. **`--noise_num_samples` default 20 -> 200.** Not for the bias (the estimator fixes
   that) but for variance: at M=20, `"higher"` is essentially the single max draw, so
   the per-pair floor is jumpy. M=200 gives a stable interior order statistic and
   resolves the a/2 net-band tails (unrepresentable below M~40). Output files become
   `*_200_metrics.json`, non-clobbering; downstream pickers prefer the higher-M file.

3. **Pruned the Delta_Topo decomposition noise floors** (overlap / non_overlap /
   discovery / abandonment / net_hi / net_lo) from the producer and their dashed bands
   in `utils/plotting_utils.py`. The decomposition is descriptive-only (paper says the
   mode diagnoses rest on PP and the aggregate metrics, not the component split), so it
   carries no floor. The decomposition **raw** values are kept (still plotted).
   `zmax_p90/p99` kept as spare percentiles.

4. **Speed #1: removed the nested `ProcessPoolExecutor`.** The two null loops called
   `compare_checkpoints` for a single pair, which spun up a whole process pool per
   split, M times, inside a worker already in the outer pool. Replaced with a direct
   `policy_comparison_worker` call. Benchmarked ~500x less spawn overhead on Windows
   (M=200: 146s -> 0.27s of pure pool overhead per pair), bit-identical output.

5. **Speed #2: precompute per-trajectory summaries once, merge per resample** instead
   of rebuilding an `EmpiricalPolicy` on every split. Needs `convert_to_hashable`
   (trajectory.py) and `compute_ngram_wasserstein_from_counts` (utils/metrics.py).
   Verified numerically equivalent.

## The empirical A/B (the decisive result)

`noise_null_ab.py` compares, on Random runs (every fire is a false positive, so
fire-rate = FPR), the four combinations of {pooled, target-only} x {weighted,
unweighted} `strategic_shift`. Random MC/Taxi/FL trajectories, M=200.

Family-wise FPR (the actual detector decision):

| null x weighting     | Taxi  | FrozenLake | MountainCar |
|----------------------|-------|------------|-------------|
| pooled / weighted    | 5.6%  | 3.6%       | 19.5%       |
| pooled / unweighted  | 7.0%  | 3.1%       | 16.1%       |
| target / weighted    | 0.0%  | 0.0%       | 3.4%        |
| target / unweighted  | 0.0%  | 7.3%       | 16.1%       |

Conclusions:

- **Pooled / weighted is correct.** Calibrated on Taxi (5.6%) and FrozenLake (3.6%).
- **Target-only is rejected.** 0.0% on Taxi and FrozenLake means the floor is so
  over-inflated that nothing clears it: zero sensitivity on real learners. It only
  helped MountainCar. Cause: on low-diversity / near-deterministic eval, splitting a
  single checkpoint's N trajectories into N/2 halves inflates the floor badly.
- **Removing the weighting is rejected.** Never systematically helps.
- **MountainCar is the lone outlier.** ~20% pooled/weighted, and it is flat across
  M=100/200/500 (higher M does NOT fix it). Driver is `strategic_shift` (~20%), not
  wasserstein (~6%, calibrated). It happens at practically-zero magnitudes: raw
  `strategic_shift` ~= 0.0006 on Random vs ~0.3-0.5 for a real change (~500x below
  signal), amplified into significance by a tiny `null_std` (~0.0002) in the z-score.
  So it is a statistical-vs-practical-significance artifact, not a misclassification
  risk: no Random run is ever mistaken for a learner.

## Resolution: effect-size gate (epsilon_min), adopted

Keep pooled/weighted (with `method="higher"` + M=200), and add a practical-significance
gate to the detector decision:

> fire iff (zmax > zmax_p95, statistically significant) AND (raw of the driving/argmax
> metric > epsilon_min[that metric], practically significant).

**epsilon_min = per-env, per-metric 99th percentile of Random raw** (calibrate on the
non-learning baseline):

| env         | Delta_Topo | Delta_Strat | Delta_Seq (wass) |
|-------------|------------|-------------|------------------|
| MountainCar | 0.0034     | 0.0009      | 0.0172           |
| Taxi        | 0.0158     | 0.0043      | 0.0221           |
| FrozenLake  | 0.0060     | 0.0353      | 0.0566           |

(topo/strat are JSD in [0,1]; wass is the un-normalized EMD in [0,3].)

The gate is applied PER METRIC: fire iff ANY decision metric is significant AND its raw
change exceeds epsilon_min. (Do NOT gate only the largest-z metric: that wrongly silences
a pair whose real, significant change sits in a different metric than the z-argmax.)

Result:
- **MountainCar Random family-wise FPR: 20.7% -> 2.3% in-sample, 3.4% held-out**
  (leave-one-seed-out CV). Report the held-out 3.4%; in-sample resubstitution is optimistic.
- **Sensitivity 100% preserved:** Taxi Good keeps 14/14 learning-phase fires, MC Good
  keeps 29/29. Nothing real is suppressed (real learning is ~0.3-0.7, far above epsilon_min).
- Post-hoc on the existing `_200_` files (no re-run): the gate ANDs the stored
  `zmax > zmax_p95` decision with `raw > epsilon_min`, so it only ever removes false fires.

Why per-env (not one global epsilon_min): a single global value FAILS. FrozenLake Random
stays ~100% FPR until eps~=0.05 (FL Random median Delta_Strat ~= 0.033 -- a stochastic
policy on slippery ice churns action distributions a lot between checkpoints), and eps=0.05
then costs sensitivity (MC Good 29->28). Per-env Random baselines differ ~40x (FL strat
0.035 vs MC strat 0.0009), so per-env-per-metric epsilon_min reflects real structure, not
overfitting.

FL RISK CLOSED (Q-learning Good runs tested, 6 seeds each). Sensitivity (fraction of
learning-phase fires that clear epsilon_min):

| env         | sensitivity                          |
|-------------|--------------------------------------|
| Taxi        | 100% (seed1)                         |
| MountainCar | 100% (6 seeds x bins_15 AND bins_30) |
| FrozenLake  | 97% (6 seeds)                        |

FL Q converges on slippery (return 0 -> ~0.65 all 6 seeds). FL's 3% loss is the weakest
learning steps at Delta_Strat ~= 0.032, right at FL's random baseline of 0.033 (i.e.
near-indistinguishable from noise); 97% is a conservative lower bound (those pairs may
not pass the significance test either, in which case the gate changed nothing).
epsilon_min[MC], calibrated on the random MC runs, transfers cleanly to BOTH MC
discretizations (bins_15, bins_30) because MC learning (0.05-0.7) sits far above it.

VERDICT: the epsilon_min gate is validated across all three environments -- FPR
controlled (MC 3.4% held-out; Taxi 5.6% / FL 3.6% significance-only, gate lowers further)
and sensitivity preserved (>=97% everywhere). Not yet reflected in the paper
(`sn-article.tex`).

Why 99th (not 0%): you cannot force Random to 0% via the significance level (that is
alpha=0, i.e. no power). The effect-size gate instead exploits the magnitude gap
(Random ~0.06 vs learning ~0.3-0.7), pushing Random to ~1-2% at zero sensitivity cost.
95th pct -> ~5%, 99th -> ~2%.

Equivalence: flooring the z-denominator, `z = (raw-mu)/max(sigma, sigma_floor)`, is the
same fix in the z-framework, with epsilon_min ~= zmax_p95 * sigma_floor. Justify sigma_floor
as variance shrinkage / a moderated statistic guarding against degenerate (near-zero)
null variances. Note: sigma_floor changes zmax_p95 (needs the null draws) so it is NOT
post-hoc; epsilon_min is. They give near-identical decisions.

## Silencing is a SEPARATE question (do not conflate with FPR)

"Converged learner never goes quiet" is NOT fixed by the Random-calibrated gate, and
should not be. Random is the wrong reference for convergence: a converged *greedy*
learner still jitters ~0.06 in Delta_Strat between checkpoints, vs a *stochastic random*
policy's ~0.004 (~15x), so the converged residual sails over any Random-calibrated
epsilon_min. The converged-tail blip is the inherent ~5% of the learner's own
significance test (it is stationary-vs-itself). On Taxi Good the detector silences 4/5
plateau pairs with one residual blip (cp 5000). **Decision: accept the residual FPs**
(the detector "diminishes at convergence"; it does not "go fully silent"). Forcing full
silence would need a larger, learner-tail-calibrated threshold and would cost
weak-signal sensitivity.

Not yet reflected in the paper (`sn-article.tex`).

## Reproduce

```
# per-env A/B (Random => FPR); needs the raw trajectory jsonl dirs
python -m experiments.noise_null_ab --traj_root <MC Random trajectories>  --num_actions 3 --M 200
python -m experiments.noise_null_ab --traj_root <Taxi Random trajectories> --num_actions 6 --M 200
python -m experiments.noise_null_ab --traj_root <FL Random trajectories>   --num_actions 4 --M 200
# add --validate <a stored *_metrics.json> to check observed values match to ~1e-16
```

Data used: MC metrics `random_mc_runs/m_{100,200,500}`; raw trajectories
`random_{mc,taxi,fl}_runs/trajectories/seed_*` (jsonl, `{"trajectory": {"states",
"actions", ...}}`). num_actions: MC 3, Taxi 6, FrozenLake 4.

## Environment note

The harness is self-contained (parses jsonl directly, imports only `utils.metrics`) to
avoid the `utils.data_utils -> trajectory_utils -> aidojo_utils -> netsecgame` import
chain. The toolkit's real runtime env needs `ot` (POT), `ruptures`, `scikit-learn`,
`networkx`, and `netsecgame`; `requirements.txt` is stale (lists numpy 2.0.2, omits
ot/scipy). On a bare Python with numpy, install `POT ruptures scikit-learn networkx`.
