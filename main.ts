// Sokoban solver for make.fun's sokoban-solver-01-run-20261121.
//
// Strategy: every level is a hallway with 1-4 rooms off it, boxes never
// leave their room, and the level's push-optimal count P* is the sum of its
// rooms' optima. So the solver decomposes a level into rooms and solves
// each room independently with a push-optimal search, then concatenates
// the rooms' move strings (the player walks on from where the last room
// left it). One push over a room's optimum halves that instance's points,
// so optimality per room is everything.
//
// Per room the search is A* over pushes (not moves): a state is the boxes'
// squares plus the player's reachable region (named by its lowest square).
// The heuristic is the minimum assignment cost of boxes to goals, where
// the cost of one box to one goal is the fewest pushes that box needs
// ignoring the other boxes (a reverse pull-back BFS from the goals). Each
// push moves one box one square, so the assignment cost can drop by at
// most one per push: the heuristic is admissible (and consistent), so the
// first solved state A* pops is push-optimal. Deadlocks are pruned with
// the same two tests the level generator relies on: a box on a square
// from which no pull-back reaches any goal, and a frozen 2x2 block of
// walls and boxes with a box off its goal. A plain push-BFS with the same
// pruning is kept as a fallback if A* ever exhausts its node cap.
//
// Determinism: every budget is counted in work (states expanded), never
// in wall-clock time, and every tie is broken by a fixed order plus an
// insertion sequence number, so two runs write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four moves: [dx, dy, step letter, push letter]. Fixed order everywhere.
const DIRS: Array<[number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// Work budgets (states expanded), never the clock.
const ASTAR_PER_ROOM = 400_000;
const BFS_PER_ROOM = 300_000;
const GLOBAL_BUDGET = 3_000_000;

let spent = 0;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface RoomSolution {
  pushes: number;
  moves: string;
}

/** Parse XSB text. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * For each goal, the fewest pushes a lone box needs to reach it from every
 * square, found by pulling the box backwards from the goal (a box at t can
 * have come from t - d when the player square t - 2d is open floor). Other
 * boxes are ignored, so every distance is a lower bound on the true cost.
 * A square with no finite distance to any goal is a dead square: a box
 * pushed there can never reach a goal.
 */
function pushDistances(level: Level): { dist: Float64Array[]; live: Uint8Array } {
  const { w, wall, goals } = level;
  const size = wall.length;
  const live = new Uint8Array(size);
  const dist: Float64Array[] = [];
  for (const g of goals) {
    const d = new Float64Array(size).fill(Infinity);
    d[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const t = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const dv = dy * w + dx;
        const s = t - dv;
        if (wall[s] || wall[s - dv]) continue;
        if (d[s] !== Infinity) continue;
        d[s] = d[t]! + 1;
        queue.push(s);
      }
    }
    for (let i = 0; i < size; i++) if (d[i] !== Infinity) live[i] = 1;
    dist.push(d);
  }
  return { dist, live };
}

/** All orderings of 0..n-1, for the assignment heuristic (n <= 4 per room). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

/** Flood-fill the squares the player reaches from `from` around walls and
 * boxes; marks them with `stamp` in `seen` and returns the lowest one. */
function flood(
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
  w: number,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
 * and boxes holding a box off its goal. */
function frozenBlock(
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
  w: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b])) return true;
    if ((occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower
 * LURD letters (first direction first on ties). */
function walkTo(
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
  w: number,
): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The player's square after following `moves` (case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

const keyOf = (boxes: number[], low: number): string => low + ':' + boxes.join(',');

/**
 * Push-optimal A* for one (sub-)level. `cap` bounds expansions. Returns
 * null when the cap runs out or the level has no solution.
 */
function solveAStar(level: Level, cap: number): RoomSolution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const { dist, live } = pushDistances(level);
  const n = level.boxes.length;
  const perms = permutations(n);
  const hOf = (boxes: number[]): number => {
    let best = Infinity;
    for (const p of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = dist[p[i]]![boxes[i]]!;
        if (d === Infinity) {
          s = Infinity;
          break;
        }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  };

  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stamp = 0;

  // States: boxes, player's square after the push that made them, parent,
  // and the push itself (from, to, direction). The open list is a heap on
  // (f, -g, sequence): lowest f first, deepest first on ties, then oldest.
  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sLow: number[] = [0];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sDir: number[] = [-1];
  const sG: number[] = [0];
  for (const b of level.boxes) occupied[b] = 1;
  sLow[0] = flood(wall, occupied, level.player, seen, ++stamp, stack, w);
  for (const b of level.boxes) occupied[b] = 0;
  const best = new Map<string, number>();
  best.set(keyOf(level.boxes, sLow[0]!), 0);
  const h0 = hOf(level.boxes);
  if (h0 === Infinity) return null;
  if (level.boxes.every((b) => goal[b])) return { pushes: 0, moves: '' };

  type HeapEntry = [number, number, number, number]; // f, g, seq, state
  const heap: HeapEntry[] = [];
  let seq = 0;
  const less = (a: HeapEntry, b: HeapEntry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, 0, seq++, 0]);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && expanded < cap) {
    const [, g, , si] = heapPop();
    const boxes = sBoxes[si]!;
    if (best.get(keyOf(boxes, sLow[si]!)) !== g) continue;
    if (boxes.every((b) => goal[b])) {
      found = si;
      break;
    }
    expanded++;
    spent++;
    for (const b of boxes) occupied[b] = 1;
    flood(wall, occupied, sPlayer[si]!, seen, ++stamp, stack, w);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(wall, occupied, goal, t, w);
        let key = '';
        let h = Infinity;
        let low = -1;
        if (!frozen) {
          low = flood(wall, occupied, b, mark, ++stamp, stack, w);
          const next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, low);
          const prev = best.get(key);
          if (prev === undefined || prev > g + 1) {
            h = hOf(next);
            if (h !== Infinity) {
              best.set(key, g + 1);
              const ni = sBoxes.length;
              sBoxes.push(next);
              sPlayer.push(b);
              sLow.push(low);
              sParent.push(si);
              sFrom.push(b);
              sTo.push(t);
              sDir.push(di);
              sG.push(g + 1);
              heapPush([g + 1 + h, g + 1, seq++, ni]);
            }
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk to each push square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = sFrom[s]!;
    const [dx, dy, , pushLetter] = DIRS[sDir[s]!]!;
    const d = dy * w + dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(wall, occupied, player, from - d, w);
    for (const b of boxes) occupied[b] = 0;
    moves += pushLetter;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/**
 * Plain push-BFS with the same pruning, used when A* exhausts its cap.
 * Expands states in push order, so the first solved state is optimal.
 */
function solveBfs(level: Level, cap: number): RoomSolution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const { live } = pushDistances(level);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stamp = 0;

  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sLow: number[] = [0];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sDir: number[] = [-1];
  for (const b of level.boxes) occupied[b] = 1;
  sLow[0] = flood(wall, occupied, level.player, seen, ++stamp, stack, w);
  for (const b of level.boxes) occupied[b] = 0;
  const index = new Map<string, number>();
  index.set(keyOf(level.boxes, sLow[0]!), 0);
  if (level.boxes.some((b) => !live[b])) return null;
  if (level.boxes.every((b) => goal[b])) return { pushes: 0, moves: '' };

  let found = -1;
  for (let head = 0; found < 0 && head < sBoxes.length; head++) {
    if (sBoxes.length >= cap) return null;
    spent++;
    const boxes = sBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    flood(wall, occupied, sPlayer[head]!, seen, ++stamp, stack, w);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(wall, occupied, goal, t, w);
        if (!frozen) {
          const next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const low = flood(wall, occupied, b, mark, ++stamp, stack, w);
          const key = keyOf(next, low);
          if (!index.has(key)) {
            if (sBoxes.length >= cap) {
              for (const x of boxes) occupied[x] = 0;
              return null;
            }
            index.set(key, sBoxes.length);
            sBoxes.push(next);
            sPlayer.push(b);
            sLow.push(low);
            sParent.push(head);
            sFrom.push(b);
            sTo.push(t);
            sDir.push(di);
            if (next.every((x) => goal[x])) {
              found = sBoxes.length - 1;
              break;
            }
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = sFrom[s]!;
    const [dx, dy, , pushLetter] = DIRS[sDir[s]!]!;
    const d = dy * w + dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(wall, occupied, player, from - d, w);
    for (const b of boxes) occupied[b] = 0;
    moves += pushLetter;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** One room's solution: A* first, plain BFS if it gives up. */
function solveRoom(level: Level): RoomSolution | null {
  const remaining = GLOBAL_BUDGET - spent;
  if (remaining <= 0) return null;
  const a = solveAStar(level, Math.min(ASTAR_PER_ROOM, remaining));
  if (a !== null) return a;
  const left = GLOBAL_BUDGET - spent;
  if (left <= 0) return null;
  return solveBfs(level, Math.min(BFS_PER_ROOM, left));
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the
 * hallway: a box there reaches no goal) out of the floor; each remaining
 * part holding a box is a room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const { live } = pushDistances(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve a level room by room; the rooms' optima add up to the level's. */
function solveLevel(level: Level): string | null {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solveRoom(level);
    return whole === null ? null : whole.moves;
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solveRoom(sub);
    if (solved === null) return null;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const text = readFileSync(join(I, id, 'level.txt'), 'utf8');
  const moves = solveLevel(parseLevel(text)) ?? '';
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
