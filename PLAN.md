# Release plan

Winnow 0.1 has these parts:

- Score and prune OLMoE models.
- Score and prune Qwen3.5 or Qwen3.6 MoE text models.
- Use a decimal value for `--keep`.
- Load calibration data from Hugging Face.
- Support channel pruning and whole-expert REAP.
- Write a Hugging Face checkpoint and full pruning data.
- Provide a reference runtime and an optional fused runtime.
- Do not include training or healing.

Before each release:

1. Run all unit tests and the CUDA parity test.
2. Build the wheel and source archive.
3. Test a clean install.
4. Prune and load one model from each supported family.
5. Publish the model checkpoints on Hugging Face.
