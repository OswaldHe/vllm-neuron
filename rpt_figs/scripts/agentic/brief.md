# Brief — Figure 10: the agentic optimization flow

**What.** A blog figure for the last section, "Agentic Optimization: Composition of RSI and Expert Knowledge": the
workflow we used to port and optimize DeepSeek V4.1 Flash on Trainium 3. One large outer loop with a human in it;
every stage of it runs its own small loop. Static PNG, as wide as the other blog figures (~680 pt, rendered at 6x),
labels ~8 pt to match them (not the paper column band; the width lint error is expected).

**What is drawn.** Two horizontal lanes, **Human** (top, narrow) and **Agent** (bottom), with the stages left to right.
Repeated element: a small circular-arrow glyph with a 2–4 word label on every stage that runs its own loop.
What differs: human-owned steps vs agent loops (colour + person icon), and the data flywheel between subkernels.

- Inputs (Human lane, far left): **Model spec** (checkpoint, tech report, reference code) and **Correctness
  constraint** (the verifier: recalled needle, trivia answer, integration test).
- ① **Correctness loop** (Agent) — implement ↻ verify until the verifier passes.
- ② **Profile** (Agent) — device and host traces of the live server.
- ③ **Grilling loop** (spans both lanes) — agent ⇄ human, back and forth until the issues and the bottleneck are
  understood.
- ④ **New constraints** (Human) — the next goal and its rules (e.g. a TPOT target, pure NKI, bit-exact).
- ⑤ **Subkernel workspaces** (Agent) — one workspace per subkernel, with **golden inputs/outputs traced** from the
  live server.
- ⑥ **RSI loops, one subkernel at a time** (Agent) — K1 → K2 → K3 → … ; inside each: edit → verify on golden I/O →
  measure → review. Each new loop starts by applying the optimizations of the previous kernel (**data flywheel**,
  arrows between the K boxes).
- ⑦ **Integrate & tune** (Agent) — put the kernels into the model and tune until the end-to-end gain matches what
  the subkernel speedups predict (integrate ↻ bench).
- Return arc ⑦ → ② under the Agent lane: **resume the loop** (the heaviest line in the figure).

**The story.** Read from the inputs: ① makes the model correct, ② profiles it, ③ agent and human work out what is
slow, ④ the human turns that into constraints, ⑤–⑥ the agent optimizes subkernel by subkernel in isolated
workspaces, carrying optimizations forward, ⑦ integrates and tunes, and the big arc returns to ②.

**Point.** The human steers the outer loop (understanding and constraints); the agent runs the inner loops
autonomously, and what it learns on one subkernel seeds the next.

**Layout and style.** Mechanism / walkthrough on a swimlane base, left-to-right, circled step numbers ①–⑦.
Palette of the other blog figures: Agent loops = purple (the "Agentic Opt" colour of Figure 1, the one accent),
human steps = neutral dark grey with a person icon, inputs = pale grey; dashed purple = data flywheel only.
Small legend (agent loop / human step / data flywheel). Closest exemplars: asplos25-131_fig2 (draft → verify
tuning loop with a feedback arrow), osdi23-045_figtex1 (nested control loop).

**Confirm.** (a) Two lanes with a return arc, or a circular racetrack? (b) One short real example per stage, or
generic labels only? (c) Does the human own only ③ and ④, or also a review gate after ⑦?

**Caption draft.** Figure 10. Our agentic optimization flow. ① An agent ports the model until it meets the
correctness constraint. ② We profile it, ③ the agent and a human grill the profile until the bottleneck is
understood, and ④ the human sets new constraints. ⑤ The agent builds one workspace per subkernel with golden
inputs and outputs traced from the live server, ⑥ runs a recursive self-improvement (RSI) loop on each subkernel in
turn, starting each from the previous kernel's optimizations, and ⑦ integrates and tunes the kernels in the model
before the loop resumes at ②.
