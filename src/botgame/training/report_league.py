from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_matches(path: Path) -> List[dict]:
    matches = []
    if not path.exists():
        return matches
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                matches.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return matches


def _elo_from_pairwise(matches: List[dict], k: float = 16.0, iters: int = 30) -> Dict[str, float]:
    players = sorted({m["agent_a"] for m in matches} | {m["agent_b"] for m in matches})
    elo = {p: 1200.0 for p in players}
    for _ in range(iters):
        for m in matches:
            a = m["agent_a"]
            b = m["agent_b"]
            s_a = 0.5 * (float(m["outcome"]) + 1.0)
            expected_a = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
            delta = k * (s_a - expected_a)
            elo[a] += delta
            elo[b] -= delta
    return elo


def summarize_league(league_state_path: Path, matches_path: Path) -> Dict[str, object]:
    state = json.loads(league_state_path.read_text(encoding="utf-8"))
    matches = _load_matches(matches_path)
    players = state.get("players", {})

    win_stats: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for m in matches:
        a, b = m["agent_a"], m["agent_b"]
        s = 0.5 * (float(m["outcome"]) + 1.0)
        win_stats[(a, b)].append(s)
        win_stats[(b, a)].append(1.0 - s)

    win_table = {
        f"{a} vs {b}": float(np.mean(scores))
        for (a, b), scores in sorted(win_stats.items())
        if len(scores) >= 1
    }
    elo = _elo_from_pairwise(matches)

    frozen = [pid for pid, meta in players.items() if not bool(meta.get("trainable", False))]
    current_main = [pid for pid, meta in players.items() if bool(meta.get("trainable", False)) and meta.get("agent_type") == "main"]
    exploitability_proxy = {}
    for main_id in current_main:
        rates = []
        for opp in frozen:
            scores = win_stats.get((main_id, opp), [])
            if scores:
                rates.append(float(np.mean(scores)))
        exploitability_proxy[main_id] = min(rates) if rates else 0.5

    return {
        "num_players": len(players),
        "num_matches": len(matches),
        "win_rates": win_table,
        "elo": dict(sorted(elo.items(), key=lambda kv: kv[1], reverse=True)),
        "exploitability_proxy_min_winrate_vs_past": exploitability_proxy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="League report: win-rate, Elo, exploitability proxy.")
    parser.add_argument("--league-state", type=Path, default=Path("reports/league/league_state.json"))
    parser.add_argument("--matches", type=Path, default=Path("reports/league/matches.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/league/summary.json"))
    args = parser.parse_args()

    summary = summarize_league(args.league_state, args.matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"players={summary['num_players']} matches={summary['num_matches']}")
    print("Top Elo:")
    for name, rating in list(summary["elo"].items())[:10]:
        print(f"  {name:24s} {rating:8.2f}")
    print("Exploitability proxy (lower is worse):")
    for name, score in summary["exploitability_proxy_min_winrate_vs_past"].items():
        print(f"  {name:24s} {score:0.3f}")
    print(f"saved_summary={args.output}")


if __name__ == "__main__":
    main()

