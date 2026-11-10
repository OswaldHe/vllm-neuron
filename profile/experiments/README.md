# Follow-up experiments for BLOG.md (2026-11-10)

Run on the trn3.48xlarge, code `deepseek_v41_trn3_staging` @ `0a32285`
(`contrib/models/deepseek_v41_8d_trn3`), with the DLAMI venv
`aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`. `engines.py` starts / stops 8-device engines
on a chosen set of logical cores (`NEURON_VISIBLE_DEVICES`: `0-31` = devices 0-7, `32-63` = devices
8-15) and runs the bench's own `vllm bench serve` cell against them. Raw output is in `results/`.

| Script | Question | Result |
|---|---|---|
| `dual.py` | Do two 8-device engines on one instance slow each other down? (512-token cell, batch 1, 3 repetitions) | No. Alone: TPOT 3.46-3.47 ms, step 12.07-12.10 ms, TTFT 148-149 ms on either half. Other engine busy: TPOT 3.47-3.50 ms, step 12.09-12.11 ms, TTFT 148-149 ms. 96 prompts on one engine: TPOT 3.65 ms; 32 prompts on both at once: 3.73-3.75 ms. Both healthy 145 s after a common start. `results/dual_in512/dual.json`, `results/dual_in512.log` |
| `content_sweep.py spec --bench` | How does speed depend on the kind of answer? (16 chat prompts, greedy, <= 256 tokens, third pass reported) | 133 tok/s (poem, 14 % accepted) to 445 tok/s (JSON, 94 %), step 12.4-12.7 ms for all. `first_pass` in the JSON shows the one-time costs (first request after start 176 s; new prompt length 1.5-4.5 s TTFT). The run also repeats the bench's in512 / in1024 cells on this 1,280-context engine (in1024: TTFT 246 ms, TPOT 3.97 ms, ITL 12.14 ms). `results/content_spec.json`, `results/bench_spec.json` |
| `content_sweep.py nospec --nospec` | Speed without speculative decoding (`DEEPSEEK_V41_NO_SPEC_DECODE=1`) | Not measurable: the fused first decode launch does not compile for one row (`NCC_IINAR001 ... is_valid_s3d3_mm`). `results/content_nospec.log`, `results/server_nospec.log` |
| `long_answer.py` | Does speed hold over a 2,000-token answer? (2,304-context engine) | Yes: step 12.2-12.6 ms throughout; essay 189.5 tok/s, code 395.5 tok/s overall. TTFT of the ~30-token prompt on this engine: 435 ms. `results/long_answer.json` |
| `batch_sweep.py` | Speed against the number of users on a ten-sequence engine (512-token prompts) | 1 / 2 / 4 / 8 / 10 users: 75 / 64 / 58 / 53 / 56 tok/s per user, step 41-47 ms, total 69 / 113 / 195 / 338 / 405 tok/s. `results/batch/batch.json` |
| `sampled.py` | Greedy against temperature 0.7 (3 prompts, 3 runs each) | Step 24.6-25.0 ms against 12.5 ms; code 183-214 against 396 tok/s, explanation 123 against 232, story 80-88 against 168. A per-request `seed` stops the engine (`results/sampled_seed_failed.log`). `results/sampled.json` |

Re-run: `source <venv>/bin/activate; cd profile/experiments; python dual.py` (and so on). Each script
starts its own servers and stops them; nothing else may use the devices it names.
