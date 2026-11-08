# Plan: Figure 9, two sessions per trn3.48xlarge

- Canvas: 680 × 304 pt, the same width as the other blog figures. Rendered at 6x (4080 px). Inter at 9.5 / 7.5 / 6.8 pt.
- (a) Server container (dashed, as in Figure 5): x 52–628, y 8–196.
  - Engines at x 64–332 and 348–616, each with a blue two-line header.
  - Each engine has 4 linked pairs (green tint), even devices on top and NeuronLinks in purple.
  - "expert group k" labels sit under each pair.
  - The users are outside the container on the outer sides, joined to the engine by a two-way arrow.
  - Engine 2 has a dashed outline and the tag "running both at once: next measurement".
  - A key line inside the container at y 188.
- (b) HBM band at y 204–296. Bars at 11.2 pt per GiB from x 164.
  - Rows: Engram (orange) | rest (grey) | TP × EP copy.
  - The row-1 copy is red because it no longer fits ("out of memory"). The row-2 copy is green ("≈ 30 GiB: fits").
  - The 36 GiB limit is a dashed line. The axis has short ticks only, with no gridlines through the labels.
- Lint findings kept on purpose:
  - canvas.width: the blog width
  - nohead on the NeuronLinks: they are undirected
  - title.centre on the two-line engine headers
  - the copy1 label sits left of the limit line
