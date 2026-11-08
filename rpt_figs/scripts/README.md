# Scripts for the BLOG.md figures

These scripts draw the figures in `rpt_figs/` that `BLOG.md` uses. Each figure is hand-written SVG in pt units, built in Python. Chromium renders it to PNG through the FigGenie skill's renderer, with the Inter font. The two GIFs are tweened from keyframes.

Each script writes to an `out/` folder next to it, which is gitignored. Copy the result into `rpt_figs/` under the name in the table.

| Figure | `rpt_figs/` file | Script | Output in `out/` | Scale |
|---|---|---|---|---|
| 1 | `optimization_trajectory.png` | `traj.py` | `trajectory.png` | 4x |
| 2 | `ttft_trajectory.png` | `ttft.py` | `ttft_trajectory.png` | 4x |
| 4 | `trainium3_neuroncore.png` | `trn3_combo.py` (bitmaps in `src_imgs/`) | `trainium3_neuroncore.png` | 2x |
| 5 | `trn3_48xlarge_topology.png` | `topology.py` | `trn3_48xlarge_topology.png` | 6x |
| 6 | `kernel_fusion.gif` | `fusion_anim.py` | `fusion.gif` | 2.5x |
| 7 | `vllm_backend_timeline.gif` | `backend_anim.py` | `backend.gif` | 2.5x |
| 8 | `sharding_prefill_vs_decode.png` | `sharding.py` | `sharding.png` | 4x |
| 9 | `two_sessions_8dev.png` | `two_sessions/two_sessions.py` | `two_sessions/out/two_sessions.png` | 6x |
| 10 | `trace_annotated.png` | `trace_annot.py` (reads `rpt_figs/trace.png`) | `trace_annotated.png` | 2x |
| 11 | `autohelix_trn2.png` | `autohelix_trn2/ah_trn2.py` | `autohelix_trn2/out/autohelix_trn2.png` | 4x |
| 12 | `agentic_flow.png` | `agentic/agentic.py` | `agentic/out/agentic_flow.png` | 6x |

Figure 3, `dsv41_architecture.png`, has no script. It is Figure 3 of the DeepSeek-V4.1-Flash technical report, cropped from the PDF. `trace.png` is a Neuron Explorer screenshot; `trace_annot.py` uses it as input.

## Running

```bash
cd rpt_figs/scripts
python3 traj.py                    # each script defaults to the scale in the table
python3 sharding.py 4              # some take the scale as their first argument
python3 two_sessions/two_sessions.py
python3 preview_states.py fusion_anim 0 3   # render chosen GIF keyframes to out/fusion_states/
cp out/trajectory.png ../optimization_trajectory.png
```

## Requirements

- python3 with lxml, Pillow, numpy, and playwright with Chromium (`playwright install chromium`).
- The FigGenie skill (plugin `figgenie@figgenie`). `blogkit.py` imports its `_svgkit` renderer and `measure_text` from `~/.claude/plugins/cache/figgenie/figgenie/0.1.0/skills/figgenie-paper-diagram/scripts`. The renderer embeds the skill's Inter font, so the output matches on any host. Update `SKILL` in `blogkit.py` if the plugin is installed elsewhere.

## Files

- `blogkit.py`: the palette, SVG primitives, text measuring, the Chromium renderer and GIF assembly. Every script imports it.
- `anim.py`: the keyframe tween engine for the two GIFs, including the segmented progress bar.
- `preview_states.py`: renders single GIF keyframes for checking.
- `agentic/`, `two_sessions/`: `brief.md` and `plan.md` hold the design notes behind these FigGenie diagrams.
- Data:
  - The data points are typed into the scripts, with their sources in each script's docstring or comments.
  - `traj.py` and `ttft.py` take their numbers from the bench reports under `contrib/models/deepseek_v41*/test/performance/`.
  - `autohelix_trn2/ah_trn2.py` takes its numbers from the AutoHelix gate lines in the commit messages of the `deepseek_v41` branch.
