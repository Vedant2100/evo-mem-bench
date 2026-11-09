# Kaggle GPU runner for EvoMemBench

**Status:** Code and GitHub Actions integration built. It cannot launch until Kaggle credentials are configured. No actual Kaggle GPU execution has been verified.

## Scope

The first supported workload is the **hidden-rule memory diagnostic** already present in this repository, NOT the full ALFWorld/CROSSEP-EMB evaluator. The full environment needs its own ALFWorld server, game assets and compatible model-serving stack.

- The Kaggle notebook checks out the precise triggering GitHub commit SHA.
- The default local GPU model is Qwen/Qwen2.5-1.5B-Instruct, served in-process through Hugging Face Transformers (no paid API).
- Original hidden-rule prompts, evaluation conditions, memory methods and competence gate are preserved.
- Raw result JSON is stored under /kaggle/working/evomem_results.
- GitHub Actions checks status every 30 minutes, reads logs, and can collect completed JSON and text outputs as workflow artifacts.

## One-time setup

1. Sign in to Kaggle. Verify your account if required; make sure GPU and Internet-enabled notebooks are permitted. Free GPU access is quota-limited and not guaranteed.
2. Generate a personal API token at https://www.kaggle.com/settings/api. Do not paste credentials into ChatGPT or commit them.
3. In GitHub repository Settings → Secrets and variables → Actions, create **secret KAGGLE_API_TOKEN** (token string), and **variable KAGGLE_KERNEL_REF** (for example YOUR_USERNAME/evomembench-hidden-rule-gpu).
4. For first notebook creation only, set **variable KAGGLE_ALLOW_FIRST_LAUNCH** to true. Remove it or set false after the first successful launch.
5. Open Actions → Kaggle EvoMemBench GPU controller → Run workflow → launch. Start with default six worlds, one seed and four methods. A submitted notebook does not imply a completed experiment.
6. Open the notebook at https://www.kaggle.com/code/YOUR_USERNAME/evomembench-hidden-rule-gpu. Use workflow operations **status** and **collect** to inspect execution and download outputs.

## Launching from ChatGPT web

The same workflow watches changes on the main branch to kaggle/launch-request.json. By default it is **unarmed** and will not spend GPU quota. A deliberate update setting armed=true and a new distinct request_id is the explicit launch signal. Once submitted, return armed=false. Do not re-run an armed launch workflow: Kaggle cannot guarantee exactly-once runs.

The controller refuses an additional launch when a Kaggle notebook is queued or running. Status API errors fail closed. Scheduled GitHub Actions operate while ChatGPT is closed, although GitHub schedules may be delayed or disabled after extended repository inactivity.

## Limitations

- This path produces **hidden-rule diagnostic scores only**, not official EvoMemBench ALFWorld results.
- A weak 1.5B worker can fail the oracle competence test. That does not prove memory is useless.
- Kaggle has finite GPU sessions and intermittent availability. Runtime files and model caches are not durable between fresh notebook sessions. Only completed notebook output is preserved.
- The experiment must restart with fresh state after interruption; naive resume from episode files would contaminate adaptive memory conditions.
- The controller caps worlds, seeds, methods and planned model calls. It does not perform automatic retry loops.
- The Kaggle CLI must have Internet access to clone code and download Hugging Face weights. Only allowlisted open models are accepted.
- Collected GitHub Actions artifacts expire after 30 days.

References: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md and https://www.kaggle.com/docs/notebooks .
