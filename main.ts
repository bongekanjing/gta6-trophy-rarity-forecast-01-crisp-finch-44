// Forecast every GTA VI trophy's earn rate on Exophase at each case's target.
//
// The page prints a rate, but a rate is a ratio: earners divided by tracked players. Both move, and
// they move at different speeds, which is the whole of the forecast. Writing
//
//   rate_i(t) = 100 * E_i(t) / T(t)
//
// the rate falls when T grows faster than E_i. On launch night the recorded page held at 638 tracked
// players and 943 trophies earned for eight hours, then refreshed to 3,868 players with every rate
// but one lower: the new players were people who had just started, not people who had finished. The
// share of players holding each trophy therefore settles at a level of its own, and the forecast is
// two estimates, one of how far T grows by the target and one of how far each trophy's earners keep
// up with it.
//
// Both are fitted on the case's own history wherever that history holds more than one distinct
// reading. A history of one unchanged page (a page Exophase has not refreshed yet) carries no
// measurement, and a launch-week prior stands in; the prior is calibrated on the same launch-night
// dynamics the rules record, measured trophy by trophy across the refresh.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, and the standard normal's quantile at each. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6449, -0.6745, 0, 0.6745, 1.6449];

const DAY_MS = 86_400_000;

interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string };
  trophies: Array<{ id: number; name: string; tier: string; points: number; secret?: boolean }>;
  history: Read[];
}

/** The scoring clamps a rate to [0.01, 99.99] before its logit: a page printing 0% or 100% has none. */
const clamp = (pct: number): number => Math.min(99.99, Math.max(0.01, pct));
const logit = (pct: number): number => {
  const p = clamp(pct) / 100;
  return Math.log(p / (1 - p));
};
const invlogit = (z: number): number => 100 / (1 + Math.exp(-z));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Least squares y = a + b x over the points given, or null when x carries no variance. */
function fitLine(xs: readonly number[], ys: readonly number[]): { a: number; b: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i += 1) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - mx;
    num += dx * (ys[i] - my);
    den += dx * dx;
  }
  if (den <= 1e-12) return null;
  const b = num / den;
  return { a: my - b * mx, b };
}

/** The distinct tracked-player counts in a history, each with the last time it was read. */
function trackedPoints(history: readonly Read[]): Array<{ t: number; T: number }> {
  const out: Array<{ t: number; T: number }> = [];
  for (const read of history) {
    const T = read.players_tracked;
    const t = Date.parse(read.at);
    if (!Number.isFinite(T) || T <= 0 || !Number.isFinite(t)) continue;
    const prev = out[out.length - 1];
    if (prev !== undefined && prev.T === T) {
      prev.t = t;
      continue;
    }
    out.push({ t, T });
  }
  return out;
}

/**
 * How far tracked players grow from the case's last read to the target.
 *
 * Exophase's count is the population that owns the game and links the site, so it is front-loaded
 * after a launch and then slows: a power law in the age of the game, T ~ age^k, is the shape that
 * carries a launch, and it is fitted here on every distinct count the history records. Where the
 * history holds one count the page has not been refreshed and there is nothing to fit, so the
 * exponent falls back to a launch-week prior.
 *
 * The two regimes are genuinely different and worth naming. A fitted exponent over a day or two of
 * refreshes is a measurement and is trusted. A single stale page is not: the practice cases are
 * exactly that, and their answer was six times the count their page showed, because the refresh had
 * not landed when the page was read. The prior below therefore sits high enough to allow for a
 * large unreported step, without pretending to know its size.
 */
function growth(history: readonly Read[], targetMs: number, releaseMs: number): { factor: number; fitted: boolean } {
  const pts = trackedPoints(history);
  const last = pts[pts.length - 1];
  if (last === undefined) return { factor: 1, fitted: false };
  if (!Number.isFinite(targetMs) || targetMs <= last.t) return { factor: 1, fitted: pts.length > 1 };

  // Age is measured from the game's release, never from the first read: the power law describes the
  // game's life, and a first read hours into launch is already deep into its fastest growth.
  const age = (t: number): number => Math.max(0.5, (t - releaseMs) / 3_600_000);

  if (pts.length >= 3) {
    const xs = pts.map((p) => Math.log(age(p.t)));
    const ys = pts.map((p) => Math.log(p.T));
    const fit = fitLine(xs, ys);
    if (fit !== null) {
      // A launch count grows, but an exponent outside this band is noise from a quiet page.
      const k = Math.min(1.8, Math.max(0.12, fit.b));
      const ratio = age(targetMs) / age(last.t);
      return { factor: Math.max(1, Math.pow(ratio, k)), fitted: true };
    }
  }

  // One or two distinct counts: a launch-week prior. If two counts exist their slope is still a
  // better guide than nothing, since two refreshes over a day are a direct measurement of the pace.
  let k = 0.62;
  if (pts.length === 2) {
    const [a, b] = pts;
    const dx = Math.log(age(b.t)) - Math.log(age(a.t));
    if (dx > 1e-6) {
      const measured = (Math.log(b.T) - Math.log(a.T)) / dx;
      if (Number.isFinite(measured)) k = Math.min(1.8, Math.max(0.12, measured));
    }
  }
  const ratio = age(targetMs) / age(last.t);
  return { factor: Math.max(1, Math.pow(ratio, k)), fitted: pts.length === 2 };
}

/**
 * One trophy's share of the tracked population: alpha in E ~ T^alpha.
 *
 * Measured on the case's own history when it holds three or more distinct counts with a rate for
 * the trophy. Where it does not, the launch-night refresh is the guide, and it said something
 * sharply differentiated: measured trophy by trophy across that refresh, the story trophies kept
 * their share (alpha near 1: everyone who starts finishes the first chapter), the mid-game ones kept
 * a third to three quarters, and the hard, late and collection trophies kept almost nothing (alpha
 * near 0: the few hundred people who held them at 638 players are the same few hundred at 14,000).
 *
 * The ordering the prior follows is how much of the tracked population the trophy reaches, because
 * that is what sets it: a trophy everyone has cannot lose share, a trophy one player in two thousand
 * has cannot gain any. Tier adjusts it, a platinum or gold being finished late by a committed core.
 */
function alpha(history: readonly Read[], index: number, tier: string): { a: number; fitted: boolean } {
  const pts: Array<{ T: number; E: number }> = [];
  for (const read of history) {
    const r = read.rates[index];
    const T = read.players_tracked;
    if (r === null || r === undefined || !Number.isFinite(r) || r <= 0) continue;
    if (!Number.isFinite(T) || T <= 0) continue;
    const E = (clamp(r) / 100) * T;
    if (E <= 0) continue;
    const prev = pts[pts.length - 1];
    if (prev !== undefined && prev.T === T) {
      pts[pts.length - 1] = { T, E };
      continue;
    }
    pts.push({ T, E });
  }
  if (pts.length >= 3) {
    const fit = fitLine(pts.map((p) => Math.log(p.T)), pts.map((p) => Math.log(p.E)));
    if (fit !== null && Number.isFinite(fit.b)) {
      // An alpha above 1 is real early (a trophy spreading through a growing population faster than
      // the population grows) and is allowed; below 0 says earners were lost, which cannot happen.
      return { a: Math.min(1.2, Math.max(0, fit.b)), fitted: true };
    }
  }
  return { a: alphaPrior(history, index, tier), fitted: false };
}

/** The launch-night prior for alpha, keyed on how much of the tracked population the trophy reaches. */
function alphaPrior(history: readonly Read[], index: number, tier: string): number {
  const lastRates = history[history.length - 1]?.rates ?? [];
  const r = lastRates[index];
  const seen = typeof r === 'number' && Number.isFinite(r) ? clamp(r) : 0.5;
  let a: number;
  if (seen >= 60) a = 1.0;
  else if (seen >= 20) a = 0.85;
  else if (seen >= 8) a = 0.62;
  else if (seen >= 3) a = 0.4;
  else if (seen >= 1) a = 0.18;
  else if (seen >= 0.3) a = 0.07;
  else a = 0.03;
  if (tier === 'platinum') a *= 0.55;
  else if (tier === 'gold') a *= 0.75;
  return Math.min(1, Math.max(0, a));
}

/**
 * The spread of a trophy's log-rate at the target, in logit units.
 *
 * The forecast's error is the growth factor's error, and it enters each trophy multiplied by
 * (1 - alpha): a trophy whose earners keep pace is not exposed to the count at all, while one whose
 * earners are fixed is exposed to all of it. On top sits the trophy's own error in keeping pace,
 * which is small when the case measured alpha and larger when the prior stands in.
 *
 * The width is not a style choice: pinball loss is minimised only by the true quantiles, so an
 * honest spread is also the highest-scoring one. The practice is the evidence for how wide honest is
 * here. The house baseline used 0.5 per root-day of horizon, which over its 90 minutes is 0.125, and
 * its 90% band covered none of the rates it was scored on, because a refresh it had never seen moved
 * them by whole logit units.
 */
function spread(a: number, fitted: boolean, horizonDays: number): number {
  // How uncertain the tracked count is at the target, in natural log. A week out, a launch count can
  // plausibly land half or double a fitted curve; 0.5 in log is about that range.
  const tSd = 0.5;
  const exposure = 1 - a;
  // A trophy's own pace: measured alpha is worth much more than an assumed one.
  const own = fitted ? 0.14 : 0.3;
  // The horizon widens both: a longer forecast is a longer extrapolation.
  const widen = 0.8 + 0.2 * Math.min(2, Math.sqrt(Math.max(0.25, horizonDays)));
  return Math.sqrt(Math.pow(tSd * exposure, 2) + own * own) * widen;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const files = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();

for (const file of files) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;

  const targetMs = Date.parse(c.target);
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const horizonDays = c.horizon_minutes / 1440;

  const g = growth(c.history, targetMs, Number.isFinite(releaseMs) ? releaseMs : targetMs - 7 * DAY_MS);
  const logFactor = Math.log(g.factor);

  const rows = c.trophies.map((t, i) => {
    // The last read that actually lists this trophy: a read can lack it, and an older rate is a
    // worse guide than none, so the newest one wins.
    let base: number | null = null;
    for (let j = c.history.length - 1; j >= 0; j -= 1) {
      const r = c.history[j].rates[i];
      if (r !== null && r !== undefined && Number.isFinite(r)) {
        base = r;
        break;
      }
    }
    if (base === null) base = 0.05;

    const al = alpha(c.history, i, t.tier);
    // rate_target = rate_last * factor ** (alpha - 1): an alpha of 1 holds the rate, an alpha of 0
    // divides it by the whole growth of the tracked population.
    const medianRate = base * Math.exp((al.a - 1) * logFactor);
    const m = logit(medianRate);
    const s = spread(al.a, al.fitted, horizonDays);

    const q = Z.map((z) => round4(100 / (1 + Math.exp(-(m + z * s)))));
    // The row must be non-decreasing and inside the page's own range; rounding can tie, never cross.
    for (let k = 1; k < q.length; k += 1) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    return { id: t.id, q };
  });

  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
