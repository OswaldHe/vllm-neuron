# Brief — figure for "Serving Two High Interactivity Sessions at a Time"

**What.** This is a blog figure for that section of BLOG.md. It becomes Figure 9, and the trace and agentic figures are renumbered to 10 and 11. It is a static PNG, as wide as Figures 5 and 10 (680 pt), rendered at 6x. Labels are 6.8–8.5 pt, Inter, in the blog palette (blogkit).

**Kind.** Composite: (a) a scope/architecture panel, the instance split into two engines; (b) a data-to-scale panel, the HBM of one logical core.

**(a) One trn3.48xlarge, two independent engines.**
- The server container is dashed, as in Figure 5.
- Two halves sit side by side:
  - **Engine 1**: devices 0–7, cores 0–31.
  - **Engine 2**: devices 8–15, cores 32–63.
- Each half has:
  - a vLLM engine header (blue, the backend colour): TP = 32, `NEURON_VISIBLE_DEVICES`, and its own collectives port
  - its 8 devices drawn as Figure 5 does: 4 linked pairs, even devices on top
  - each pair tinted green: one attention TP8 group, which also holds one 96-expert group of the TP8 × EP4 decode MoE
- A user on the outer side of each half, with a two-way arrow to the engine: "one user, ~288 tok/s".
- Engine 2 has a dashed outline and the tag "running both at once: next measurement". Only devices 0–7 have been measured.

**(b) Fitting one engine into a logical core's 36 GiB HBM.**
- Two stacked bars, to scale, with a dashed 36 GiB limit line:
  1. **As ported from 16 devices.** Engram tables BF16 2 × 5.72 = 11.4 GiB, plus the rest of the weights and buffers (per-op FP8 copies kept), make 28.7 GiB in use. The TP × EP copy of the decode MoE adds 9.4 GiB, which runs past 36: out of memory (red).
  2. **Engram FP8, per-op copies freed.** Engram 2 × 2.95 = 5.9 GiB plus the rest make 20.4 GiB. With the 9.4 GiB copy that is ≈ 30 GiB, which fits.
- Segment sizes come from the README, the memory notes and the worker logs. The "rest" segment is the documented total minus the Engram tables: 17.3 GiB, then 14.5 GiB.

**Point.** At batch 1, half the instance is enough for one high-interactivity user. The other half is free for a second engine, and the main thing that had to change to make it fit was memory.

**Not drawn.** The 6 → 4 NEFF executions; Figure 6 already shows the merged launches.

**Assumptions** (stated to the user at delivery):
- Expert groups are labelled "group 1–4", not expert ids.
- The concurrent second engine is shown as planned.
- The rest-of-weights segment is derived.
- The cost numbers stay in the table, not the figure.

**Caption draft.** Figure 9. (a) One trn3.48xlarge split into two independent 8-device engines, each serving one user. Each engine runs TP = 32 on its own devices and collectives port. Each linked device pair (green) holds one attention TP8 copy and one group of 96 routed experts of the TP8 × EP4 decode MoE. We have measured engine 1; running both at once is our next measurement. (b) HBM of one logical core, to scale. As first ported, the TP × EP copy of the decode MoE no longer fit. Keeping the Engram tables in FP8 and freeing the per-op FP8 weights left ~6 GiB of headroom.
