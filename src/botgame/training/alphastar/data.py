from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from botgame.common.types import Action
from botgame.training.alphastar.action_codec import FactorizedActionCodec
from botgame.training.alphastar.features import live_event_to_features, observation_to_features
from botgame.training.alphastar.types import TrajectorySequence

_MOVE_TOKEN_RE = re.compile(r"\bKey[WASD]\b", flags=re.IGNORECASE)
_ABILITY_KEY_TO_ID = {
    "digit1": 1,
    "digit2": 2,
    "digit3": 3,
    "keyr": 4,
}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _action_from_string(action_name: str) -> Action:
    lower = action_name.lower()
    fire = "attack" in lower or "fire" in lower
    interact = "interact" in lower or "loot" in lower
    move_x = 0.0
    move_y = 0.0
    if "left" in lower:
        move_x = -1.0
    elif "right" in lower:
        move_x = 1.0
    if "up" in lower:
        move_y = -1.0
    elif "down" in lower:
        move_y = 1.0
    aim_x = 1.0 if fire else 0.0
    return Action(
        move_x=move_x,
        move_y=move_y,
        aim_x=aim_x,
        aim_y=0.0,
        fire=fire,
        ability_id=None,
        interact=interact,
    )


def _extract_key_tokens(action_name: str, active_keys: Iterable[Any] | None) -> set[str]:
    keys: set[str] = set()
    if action_name:
        keys.update(token.upper() for token in _MOVE_TOKEN_RE.findall(action_name))
    if active_keys:
        for key in active_keys:
            try:
                key_s = str(key).strip()
            except Exception:
                continue
            if not key_s:
                continue
            upper = key_s.upper()
            if upper in {"KEYW", "KEYA", "KEYS", "KEYD"}:
                keys.add(upper)
    return keys


def _ability_from_live_event(action_name: str, active_keys: Iterable[Any] | None, ability_last: Any) -> int | None:
    candidates: List[str] = []
    try:
        ability_last_s = str(ability_last or "").strip()
    except Exception:
        ability_last_s = ""
    if ability_last_s:
        candidates.append(ability_last_s)
    if active_keys:
        for key in active_keys:
            try:
                key_s = str(key).strip()
            except Exception:
                continue
            if key_s:
                candidates.append(key_s)
    if action_name:
        for key_s in ("Digit1", "Digit2", "Digit3", "KeyR"):
            if key_s.lower() in action_name.lower():
                candidates.append(key_s)

    for key_s in candidates:
        normalized = str(key_s or "").strip().lower()
        if normalized in _ABILITY_KEY_TO_ID:
            return _ABILITY_KEY_TO_ID[normalized]
    return None


def _action_from_live_event(event: dict[str, Any]) -> Action:
    action_name = str(event.get("action", "") or "")
    active_keys = event.get("active_keys", []) or []
    key_tokens = _extract_key_tokens(action_name, active_keys)

    move_x = 0.0
    move_y = 0.0
    if "KEYA" in key_tokens:
        move_x -= 1.0
    if "KEYD" in key_tokens:
        move_x += 1.0
    if "KEYW" in key_tokens:
        move_y -= 1.0
    if "KEYS" in key_tokens:
        move_y += 1.0

    if move_x == 0.0 and move_y == 0.0:
        fallback = _action_from_string(action_name)
        move_x = fallback.move_x
        move_y = fallback.move_y

    action_lower = action_name.lower()
    fire = (
        ("attack" in action_lower)
        or ("fire" in action_lower)
        or any(str(k).strip().lower() == "mouseright" for k in active_keys)
    )
    interact = (
        ("interact" in action_lower)
        or ("loot" in action_lower)
        or ("chest" in action_lower)
        or ("cofre" in action_lower)
    )
    ability_id = _ability_from_live_event(
        action_name=action_name,
        active_keys=active_keys,
        ability_last=event.get("ability_last"),
    )
    aim_x = 1.0 if fire else 0.0
    return Action(
        move_x=float(np.clip(move_x, -1.0, 1.0)),
        move_y=float(np.clip(move_y, -1.0, 1.0)),
        aim_x=aim_x,
        aim_y=0.0,
        fire=bool(fire),
        ability_id=ability_id,
        interact=bool(interact),
    )


def _terminal_outcome_from_run_finish(run_finish: str | None) -> float:
    if not run_finish:
        return 0.0
    txt = run_finish.lower()
    if "win" in txt:
        return 1.0
    if "loss" in txt or "death_event" in txt or "defeat" in txt:
        return -1.0
    if "draw" in txt:
        return 0.0
    return 0.0


def _terminal_outcome_from_session_end(session_end_event: dict[str, Any] | None) -> float:
    if not session_end_event:
        return 0.0
    outcome = _terminal_outcome_from_run_finish(str(session_end_event.get("run_stop_reason", "") or ""))
    if outcome != 0.0:
        return outcome
    death_info = session_end_event.get("death", {}) or {}
    if bool(death_info.get("active", False)):
        return -1.0
    return 0.0


def load_manifest_outcomes(reports_live_dir: Path) -> Dict[str, float]:
    """Builds runtime_feedback-path -> terminal outcome mapping from live manifests."""
    outcome_map: Dict[str, float] = {}
    for manifest in reports_live_dir.glob("**/manifest.jsonl"):
        for record in _read_jsonl(manifest):
            runtime_feedback = record.get("runtime_feedback")
            if not runtime_feedback:
                continue
            outcome_map[str(Path(runtime_feedback).as_posix())] = _terminal_outcome_from_run_finish(
                str(record.get("run_finish", "") or "")
            )
    return outcome_map


def load_botgame_sequences(
    data_dir: Path,
    codec: FactorizedActionCodec,
    default_behavior_logp: float = 0.0,
) -> List[TrajectorySequence]:
    sequences: List[TrajectorySequence] = []
    for file_path in sorted(data_dir.glob("*.jsonl.gz")):
        obs: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[float] = []
        behavior_logp: List[float] = []
        actions: Dict[str, List[int]] = {name: [] for name in codec.head_sizes}
        for transition in _read_jsonl_gz(file_path):
            observation = transition.get("observation", {}) or {}
            action_dict = transition.get("action", {}) or {}
            encoded = codec.encode_action(action_dict)
            obs.append(observation_to_features(observation))
            rewards.append(float(transition.get("reward", 0.0) or 0.0))
            dones.append(float(bool(transition.get("done", False))))
            behavior_logp.append(float(transition.get("behavior_logp", default_behavior_logp) or default_behavior_logp))
            for key in actions:
                actions[key].append(int(encoded[key]))
        if not obs:
            continue
        sequences.append(
            TrajectorySequence(
                obs=np.asarray(obs, dtype=np.float32),
                actions={k: np.asarray(v, dtype=np.int64) for k, v in actions.items()},
                rewards=np.asarray(rewards, dtype=np.float32),
                dones=np.asarray(dones, dtype=np.float32),
                behavior_logp=np.asarray(behavior_logp, dtype=np.float32),
                extras={"source": "botgame", "path": str(file_path)},
            )
        )
    return sequences


def load_live_feedback_sequences(
    feedback_root: Path,
    codec: FactorizedActionCodec,
    manifest_outcomes: Dict[str, float] | None = None,
) -> List[TrajectorySequence]:
    """Converts `reports/feedback_training/**/feedback_stream.jsonl` to trajectories."""
    manifest_outcomes = manifest_outcomes or {}
    sequences: List[TrajectorySequence] = []

    for feedback_file in sorted(feedback_root.glob("**/feedback_stream.jsonl")):
        events = list(_read_jsonl(feedback_file))
        if not events:
            continue
        session_end_event = None
        for event in events:
            if str(event.get("event", "")) == "session_end":
                session_end_event = event

        obs: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[float] = []
        behavior_logp: List[float] = []
        actions: Dict[str, List[int]] = {name: [] for name in codec.head_sizes}

        for event in events:
            if event.get("event"):
                continue
            action_name = event.get("action")
            if not action_name:
                continue
            event_action = _action_from_live_event(event)
            encoded = codec.encode_action(event_action)
            obs.append(live_event_to_features(event))
            rewards.append(0.05 if bool(event.get("action_ok", False)) else -0.05)
            dones.append(0.0)
            behavior_logp.append(0.0)
            for key in actions:
                actions[key].append(int(encoded[key]))

        if len(obs) < 2:
            continue
        dones[-1] = 1.0
        normalized = str(feedback_file.as_posix())
        manifest_outcome = manifest_outcomes.get(normalized)
        session_end_outcome = _terminal_outcome_from_session_end(session_end_event)
        if manifest_outcome is None:
            terminal_outcome = session_end_outcome
        elif abs(float(manifest_outcome)) < 1e-8 and session_end_outcome != 0.0:
            terminal_outcome = session_end_outcome
        else:
            terminal_outcome = float(manifest_outcome)
        rewards[-1] += terminal_outcome
        sequences.append(
            TrajectorySequence(
                obs=np.asarray(obs, dtype=np.float32),
                actions={k: np.asarray(v, dtype=np.int64) for k, v in actions.items()},
                rewards=np.asarray(rewards, dtype=np.float32),
                dones=np.asarray(dones, dtype=np.float32),
                behavior_logp=np.asarray(behavior_logp, dtype=np.float32),
                extras={"source": "live_feedback", "path": str(feedback_file), "terminal_outcome": terminal_outcome},
            )
        )
    return sequences


def load_unified_sequences(
    data_dir: Path = Path("data/processed"),
    feedback_dir: Path = Path("reports/feedback_training"),
    reports_live_dir: Path = Path("reports/live"),
    include_live_feedback: bool = True,
    codec: FactorizedActionCodec | None = None,
) -> List[TrajectorySequence]:
    """Loads all available training sources into a single trajectory list."""
    codec = codec or FactorizedActionCodec()
    sequences = load_botgame_sequences(data_dir=data_dir, codec=codec)
    if include_live_feedback:
        manifest_outcomes = load_manifest_outcomes(reports_live_dir=reports_live_dir)
        sequences.extend(
            load_live_feedback_sequences(
                feedback_root=feedback_dir,
                codec=codec,
                manifest_outcomes=manifest_outcomes,
            )
        )
    return sequences


def dataset_sanity(
    sequences: List[TrajectorySequence],
    codec: FactorizedActionCodec,
) -> Dict[str, Any]:
    episode_lengths = [seq.length for seq in sequences]
    action_hist: Dict[str, Counter[int]] = {name: Counter() for name in codec.head_sizes}
    source_hist: Counter[str] = Counter()
    reward_sum = 0.0
    total_steps = 0

    for seq in sequences:
        total_steps += seq.length
        reward_sum += float(seq.rewards.sum())
        source_hist[str(seq.extras.get("source", "unknown"))] += 1
        for key, arr in seq.actions.items():
            action_hist[key].update(int(v) for v in arr.tolist())

    return {
        "episodes": len(sequences),
        "total_steps": total_steps,
        "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "min_episode_length": int(min(episode_lengths)) if episode_lengths else 0,
        "max_episode_length": int(max(episode_lengths)) if episode_lengths else 0,
        "reward_sum": reward_sum,
        "reward_per_step": reward_sum / max(total_steps, 1),
        "sources": dict(source_hist),
        "action_histogram": {k: dict(v) for k, v in action_hist.items()},
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified dataset loader for AlphaStar-style training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feedback-dir", type=Path, default=Path("reports/feedback_training"))
    parser.add_argument("--reports-live-dir", type=Path, default=Path("reports/live"))
    parser.add_argument("--no-live-feedback", action="store_true")
    parser.add_argument("--bins", type=int, default=11)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    codec = FactorizedActionCodec(bins=args.bins)
    sequences = load_unified_sequences(
        data_dir=args.data_dir,
        feedback_dir=args.feedback_dir,
        reports_live_dir=args.reports_live_dir,
        include_live_feedback=not args.no_live_feedback,
        codec=codec,
    )
    stats = dataset_sanity(sequences=sequences, codec=codec)
    print(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
