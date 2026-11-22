# sokoban-solver-01-run-20261121

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/awann-bit/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/awann-bit/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver for make.fun's "Build the best Sokoban
solver, 21 November 2026" bounty. Every level is a hallway with 1-4
rooms off it, boxes never leave their room, and the level's push-optimal
count is the sum of its rooms' optima — so the solver decomposes each
level into rooms and solves every room independently with A* over pushes.

The search uses an admissible heuristic (the minimum assignment cost of
boxes to goals, from per-box minimum-push distances found by pulling
boxes backwards from the goals), dead-square pruning, and frozen-2x2
block pruning, with a plain push-BFS as a fallback. All budgets are
counted in work (states expanded), never in wall-clock time, so two runs
write byte-identical answers. Scores 1,000,000 on the practice set (every
instance at its push-optimal count) and on 10 extra generated sets.

## Run it

```sh
node main.ts
```

It reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt`
for every instance in `/input`.
