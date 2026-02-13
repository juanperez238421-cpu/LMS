# LMS Bot Reverse Engineering (2026-02-12)

## Scope

Objective completed on local build at `Game/`:

1. Analyze bot-related runtime mechanics from Unity IL2CPP artifacts.
2. Reconstruct bot decision flow (movement, ability/cooldown usage, combat, zone behavior).
3. Replicate this flow in Python with test coverage.

## Evidence Sources Used

- IL2CPP metadata strings:
  - `reports/reverse_engineering/metadata_symbol_hits.txt`
- Extracted gameplay config TextAssets:
  - `reports/reverse_engineering/textassets/*.txt`
- Addressables catalog:
  - `Game/LastMageStanding_Data/StreamingAssets/aa/Windows/catalog.json`
- Derived summaries:
  - `reports/reverse_engineering/catalog_summary.json`
  - `reports/reverse_engineering/mode_rules_summary.json`

## Internal Bot Mechanics (Inferred)

### 1) Bot and AI structure

`metadata_symbol_hits.txt` contains direct AI orchestration symbols:

- `AIComponent`, `AIOpponent`, `DumbBot`, `IsSmartBot`
- `SetBots`, `numBots`, `spawnAIType`, `allowSmartBots`
- `ServerAIType`, `serverAIType`

This supports a server-authoritative AI layer with configurable bot classes/types.

### 2) Core per-tick decision pipeline

`## PLAYER_HANDLER` symbol cluster indicates a structured action pipeline:

- `ProcessInputsAndMovePlayer`
- `CheckAbilityQueue`
- `UseAbility`, `UseAbilityAtPlace`, `UseAbilityTowardsDirection`
- `CanUseAbility`, `CanUseCurrentAbility`, `HasEnoughStaminaForAbility`
- `QuickDash`, `QuickDashCoroutine`, `EndDash`, `AbilityQuickDash`
- `AutoAim`, `AutoAimAction`, `SetAutoAimTarget`, `AutoAimIfMouseClicked`
- `SetKillerID`, `GetKillerID`, `IncrementNumKills`

Inference: player/bot update loop processes movement + aim + gated ability queue + dash + kill attribution each tick.

### 3) Gameplay rule/data-driven behavior

Mode TextAssets confirm externalized runtime constraints:

- `royale_mode`: resource collection and build caps (`ownedObjectLimits.*`)
- `red_vs_blue_mode`: `globalPlayerManaRegenMultiplier`, respawn scaling, chest regeneration
- `defense`: explicit `abilityCosts.*`, `abilityStartStats.*.Cooldown`, `ownedObjectLimits.*`

Inference: bot behavior is partly constrained by mode-level costs/cooldowns/object limits loaded from data files.

### 4) Guardians, abilities, loot, toxic-zone context

Catalog and summaries confirm major runtime entities:

- Guardians/player archetypes via `Player*Prefab` (19 found).
- Ability set via `AbilitySprite1024/*` (28 found).
- Loot/chest entities (`LootPrefab`, `InGameChestPrefab*`, dropped-item prefabs).
- Toxic-zone assets (`MapPoisonPrefab`, `PoisonTrapPrefab`, `FogPrefab`).

These artifacts support implementing a knowledge base keyed by guardian, ability, cooldown, and loot context.

## Python Replication Implemented

### New policy

Created `src/botgame/bots/lms_reverse_engineered.py` with:

- `LMSReverseEngineeredBot`:
  - Priority order aligned to inferred flow:
    1. Zone survival / toxic escape
    2. Combat (retreat/engage)
    3. Loot
    4. Patrol
  - Ability queue with cooldown gating:
    - Slot 1: offense
    - Slot 2: mobility/dash
    - Slot 3: defense
  - Fallback cooldown model based on extracted mode rules.
  - Adaptive pressure tracking (`zone_pressure_streak`) from HP deltas + safe-zone state.

- Snapshot loaders:
  - `load_lms_mode_rule_snapshot(...)`
  - `load_lms_catalog_snapshot(...)`

These load the extracted data files so policy tuning remains data-driven.

### Exports

`src/botgame/bots/__init__.py` now exports:

- `LMSReverseEngineeredBot`
- `LMSModeRuleSnapshot`, `LMSCatalogSnapshot`
- `load_lms_mode_rule_snapshot`, `load_lms_catalog_snapshot`

### Tests

Added `tests/test_lms_reverse_engineered_bot.py` covering:

1. Zone escape priority over combat.
2. Offense ability queue before plain fire.
3. Low-HP retreat preferring defense ability.
4. Loot prioritization (ability drop/chest behavior).
5. Snapshot loading from reverse-engineered summaries.

Result:

- `5 passed` (new suite).

## Full Run "Until Death"

Executed deterministic simulation with this replicated policy and forced high-pressure moving toxic zone.

Output:

- `reports/reverse_engineering/replicated_bot_run_until_death.json`

Run summary:

- `bot_dead: true`
- `ticks_executed: 3024`
- `duration_sec: 100.8`
- termination: `bot_dead`

This validates end-to-end behavior under full loop execution until elimination.

## How To Adapt For Custom Bots

1. Keep mode rules external and reloadable:
   - costs, cooldowns, object limits by mode.
2. Keep action policy priority-based:
   - zone > survival > combat > economy/loot.
3. Maintain per-bot memory:
   - last HP, pressure streak, ability last-used tick, enemy recency.
4. Persist guardian/ability observations:
   - build frequency/cooldown tables for adaptive strategy.
5. Evolve from rules to learned policy:
   - use this rule-based bot as bootstrap policy for imitation/RL datasets.

## Limitations

- Unity IL2CPP native method bodies were not fully decompiled in this step.
- Current flow is a high-confidence reconstruction from symbols/config assets, not byte-perfect code lifting.
- For byte-level parity, next step is full IL2CPP dump + function-level matching against `PlayerHandler` methods.
