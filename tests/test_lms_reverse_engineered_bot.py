import math

from botgame.bots.lms_reverse_engineered import (
    LMSReverseEngineeredBot,
    load_lms_catalog_snapshot,
    load_lms_mode_rule_snapshot,
)
from botgame.common.types import Entity, Item, Observation, SelfState, ZoneState


def _obs(
    *,
    hp: float = 100.0,
    ammo: int = 10,
    cooldowns=None,
    self_pos=None,
    zone_pos=None,
    zone_radius: float = 20.0,
    is_safe: bool = True,
    entities=None,
    items=None,
    tick_id: int = 0,
) -> Observation:
    if cooldowns is None:
        cooldowns = {"fire": 0.0, "digit1": 0.0, "digit2": 0.0, "digit3": 0.0}
    if self_pos is None:
        self_pos = [0.0, 0.0]
    if zone_pos is None:
        zone_pos = [0.0, 0.0]
    if entities is None:
        entities = []
    if items is None:
        items = []

    return Observation(
        self_state=SelfState(
            position=self_pos,
            velocity=[0.0, 0.0],
            hp=hp,
            cooldowns=cooldowns,
            ammo=ammo,
        ),
        zone_state=ZoneState(position=zone_pos, radius=zone_radius, is_safe=is_safe),
        tick_id=tick_id,
        visible_entities=entities,
        visible_items=items,
    )


def test_zone_escape_priority_over_combat():
    bot = LMSReverseEngineeredBot(player_id="bot_a", seed=13)
    enemy = Entity(id="enemy", position=[2.0, 0.0], type="enemy", hp=80.0)
    obs = _obs(
        self_pos=[10.0, 10.0],
        zone_pos=[0.0, 0.0],
        is_safe=False,
        entities=[enemy],
        tick_id=100,
    )

    action = bot.act(obs)

    assert action.move_x < 0.0
    assert action.move_y < 0.0
    assert action.ability_id == 2


def test_engage_uses_offense_ability_before_plain_fire():
    bot = LMSReverseEngineeredBot(player_id="bot_b", seed=7)
    enemy = Entity(id="enemy", position=[6.0, 0.0], type="enemy", hp=80.0)
    obs = _obs(
        self_pos=[0.0, 0.0],
        is_safe=True,
        entities=[enemy],
        tick_id=10,
    )

    action = bot.act(obs)
    assert action.ability_id == 1
    assert action.fire is False
    assert action.aim_x == 1.0
    assert action.aim_y == 0.0


def test_low_hp_retreat_prefers_defense_slot():
    bot = LMSReverseEngineeredBot(player_id="bot_c", seed=21)
    enemy = Entity(id="enemy", position=[3.0, 0.0], type="enemy", hp=80.0)
    obs = _obs(
        hp=20.0,
        self_pos=[0.0, 0.0],
        is_safe=True,
        entities=[enemy],
        tick_id=20,
    )

    action = bot.act(obs)
    assert action.ability_id == 3
    assert action.move_x < 0.0
    assert math.isclose(action.aim_x, 1.0, abs_tol=1e-6)


def test_loot_priority_prefers_ability_drop():
    bot = LMSReverseEngineeredBot(player_id="bot_d", seed=11)
    chest = Item(id="chest", position=[2.5, 0.0], type="chest")
    ability_drop = Item(id="ability", position=[4.0, 0.0], type="ability_drop")
    obs = _obs(
        self_pos=[0.0, 0.0],
        is_safe=True,
        items=[chest, ability_drop],
        tick_id=30,
    )

    action = bot.act(obs)
    assert action.move_x > 0.0
    assert action.interact is False

    close_obs = _obs(
        self_pos=[3.9, 0.0],
        is_safe=True,
        items=[ability_drop],
        tick_id=31,
    )
    close_action = bot.act(close_obs)
    assert close_action.interact is True


def test_load_reverse_engineering_snapshots():
    mode_snapshot = load_lms_mode_rule_snapshot(mode_name="royale_mode")
    assert mode_snapshot.mode_name == "royale_mode"

    catalog_snapshot = load_lms_catalog_snapshot()
    assert isinstance(catalog_snapshot.player_prefabs, list)
