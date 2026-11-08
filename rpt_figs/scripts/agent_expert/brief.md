# Brief: the agent-expert heterogeneous system

**What.** Figure 13 of BLOG.md, at the end of "Composing RSI with Expert Knowledge: Agent-Expert Disaggregated Optimization". It draws optimization as a heterogeneous system: two unlike processing entities joined by an interconnect, like a CPU and an accelerator. It is a static PNG, 500 × 228 pt (narrowed at the user's request), in Inter and the blog palette, rendered at 6x. There is no title inside the frame; the user removed it.

**Components**
- **Human expert** (left; neutral dark grey, with a person icon). Four capability blocks:
  - holistic view: the whole system, across layers
  - expert knowledge: shortcuts the exploration (added at the user's request)
  - decision making: trade-offs, and what to optimize next
  - analysis of complex tasks: profiles and root causes
- **Self-improving agent** (right; purple, the "Agentic Opt" colour of Figures 1 and 12, with the loop glyph). One capability, hill-climbing on a clear objective. It is drawn as a small step chart that only climbs: kept edits are purple dots and rejected ones are grey crosses. Under it is the inner loop: edit → verify → measure → keep or revert.
- **Interconnect** (middle). Three links with circled numbers. An arrow's colour is its sender: grey from the human, purple from the agent.
  - ① **Grilling**: a two-way link drawn as two opposite arrows, like a full-duplex lane. Agent and human question each other until they share the same idea.
  - ② **Constraints**, human → agent: the goal, the rules and the correctness check, for constrained optimization.
  - ③ **Feedback**, agent → human: the results of the optimization.
- An outer dashed container, without a title.

**Point.** Each side does what it is good at. The links carry understanding (①), the problem setup (②) and the outcome (③) between them.

**Assumptions**
- The examples under each capability are taken from the blog's table.
- The hill-climb chart is schematic.
- Inserted as Figure 13 at the end of the last section of BLOG.md.

**Caption draft.** Figure N. Optimization as an agent-expert heterogeneous system. The human expert handles what needs a holistic view, decisions and analysis of complex tasks. The self-improving agent hill-climbs on a clear objective. ① Grilling syncs ideas in both directions, ② the human sends constraints that turn a goal into a constrained optimization problem, and ③ the agent sends back the results. Arrow colours show the sender.
