# Winnow

Winnow scores and removes routed experts or expert channels from MoE models.
It does not train the model after pruning.

Winnow supports OLMoE and Qwen3.5 or Qwen3.6 MoE text models.

Install the command with `uv`:

```bash
uv tool install \
  "moe-winnow @ git+https://github.com/hbfreed/winnow.git@v0.1.0"
```

Prune expert channels:

```bash
winnow prune allenai/OLMoE-1B-7B-0924 \
  --keep 0.5 \
  --calibration allenai/c4 \
  --dataset-config en \
  --output ./olmoe-winnow-50
```

Use `--strategy reap` to prune whole experts. The default strategy prunes
expert channels. `--keep` must be greater than `0` and not greater than `1`.
The REAP strategy follows the
[Cerebras REAP method](https://github.com/CerebrasResearch/reap).

Install `moe-winnow` in your model environment. Then load the output:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./olmoe-winnow-50",
    trust_remote_code=True,
)
```

The output contains `winnow.json`. This file records the source model, the
calibration data, the score, and the pruning plan.

The optional fused runtime uses the
independent [`megablocks-variable`](https://github.com/hbfreed/megablocks-variable)
project.
