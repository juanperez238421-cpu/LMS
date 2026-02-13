import pytest
import math
from botgame.common.types import Observation, Action, SelfState, Entity, Item, ZoneState
from botgame.bots.scripted_utility import ScriptedUtilityBot
from botgame.server.world import PLAYER_MAX_HP, PLAYER_MAX_AMMO, PLAYER_VISION_RADIUS

def create_dummy_observation(
    self_pos=[0.0, 0.0],
    self_hp=PLAYER_MAX_HP,
    self_ammo=PLAYER_MAX_AMMO,
    visible_enemies: list[Entity] = [],
    visible_items: list[Item] = [],
    zone_pos=[0.0, 0.0],
    zone_radius=50.0,
    tick_id=0
) -> Observation:
    self_state = SelfState(
        position=self_pos,
        velocity=[0.0, 0.0],
        hp=self_hp,
        cooldowns={"fire": 0.0},
        ammo=self_ammo
    )
    zone_state = ZoneState(position=zone_pos, radius=zone_radius, is_safe=True)
    return Observation(
        self_state=self_state,
        visible_entities=visible_enemies,
        visible_items=visible_items,
        zone_state=zone_state,
        tick_id=tick_id
    )

def test_scripted_bot_no_enemies_moves_to_zone():
    bot = ScriptedUtilityBot(player_id="test_bot", seed=42, aggressiveness=0.0, loot_bias=0.0)
    observation = create_dummy_observation(
        self_pos=[10.0, 10.0],
        zone_pos=[0.0, 0.0]
    )
    action = bot.act(observation)

    # Should move towards the zone center [0,0]
    assert action.move_x != 0 or action.move_y != 0
    assert not action.fire
    assert not action.interact
    
    # Check direction: should be towards (0,0) from (10,10)
    expected_dir_x = -10.0 / math.sqrt(200)
    expected_dir_y = -10.0 / math.sqrt(200)
    assert action.move_x == pytest.approx(expected_dir_x, abs=0.1)
    assert action.move_y == pytest.approx(expected_dir_y, abs=0.1)


def test_scripted_bot_targets_closest_enemy_and_fires():
    bot = ScriptedUtilityBot(player_id="test_bot", seed=42, aggressiveness=1.0, aim_noise_degrees=0.0)
    enemy1 = Entity(id="enemy_1", position=[10.0, 0.0], type="enemy", hp=50.0)
    enemy2 = Entity(id="enemy_2", position=[-5.0, 5.0], type="enemy", hp=50.0)
    observation = create_dummy_observation(
        self_pos=[0.0, 0.0],
        visible_enemies=[enemy1, enemy2] # enemy1 is closer
    )
    action = bot.act(observation)

    # Should aim at enemy1 ([10,0] from [0,0])
    assert action.fire
    assert action.aim_x == pytest.approx(1.0) # Normalized towards (1,0)
    assert action.aim_y == pytest.approx(0.0)
    
    # Should strafe (move perpendicular)
    assert action.move_x == pytest.approx(0.0, abs=0.1) # Expect strafing along y-axis
    assert action.move_y == pytest.approx(1.0, abs=0.1) or action.move_y == pytest.approx(-1.0, abs=0.1)


def test_scripted_bot_retreats_when_hp_low():
    bot = ScriptedUtilityBot(player_id="test_bot", seed=42, aggressiveness=0.5) # Aggressiveness 0.5 means retreat if HP < 50
    enemy = Entity(id="enemy_1", position=[10.0, 0.0], type="enemy", hp=50.0)
    observation = create_dummy_observation(
        self_pos=[0.0, 0.0],
        self_hp=40.0, # Low HP
        visible_enemies=[enemy]
    )
    action = bot.act(observation)

    # Should move away from the enemy (from [0,0] away from [10,0])
    assert action.move_x == pytest.approx(-1.0)
    assert action.move_y == pytest.approx(0.0)
    # Should still aim at enemy while retreating
    assert action.aim_x == pytest.approx(1.0)
    assert action.aim_y == pytest.approx(0.0)

def test_scripted_bot_loots_when_safe():
    bot = ScriptedUtilityBot(player_id="test_bot", seed=42, aggressiveness=1.0, loot_bias=1.0) # Prioritize loot
    health_pack = Item(id="hp_1", position=[5.0, 5.0], type="health_pack")
    observation = create_dummy_observation(
        self_pos=[0.0, 0.0],
        visible_items=[health_pack]
    )
    action = bot.act(observation)

    # Should move towards health pack
    assert action.move_x != 0 or action.move_y != 0
    assert not action.fire
    
    # When close, should interact
    observation_close = create_dummy_observation(
        self_pos=[4.5, 5.0], # Very close to health pack
        visible_items=[health_pack]
    )
    action_close = bot.act(observation_close)
    assert action_close.interact

def test_scripted_bot_reaction_delay():
    bot = ScriptedUtilityBot(player_id="test_bot", seed=42, reaction_delay_ticks=5)
    
    # First few ticks, bot should do nothing (if no last obs)
    observation_initial = create_dummy_observation(tick_id=0)
    action_initial = bot.act(observation_initial)
    assert action_initial.move_x == 0 and action_initial.move_y == 0
    assert not action_initial.fire
    
    # After delay, it should act
    # Simulate some ticks without new observation being processed
    for i in range(bot.reaction_delay_ticks):
        action = bot.act(create_dummy_observation(tick_id=i+1, self_pos=[10.0, 10.0], zone_pos=[0.0, 0.0]))
        assert action.move_x == 0 and action.move_y == 0
    
    # Now, after delay, it should act based on the last observation passed during delay
    action_after_delay = bot.act(create_dummy_observation(tick_id=bot.reaction_delay_ticks + 1, self_pos=[10.0, 10.0], zone_pos=[0.0, 0.0]))
    assert action_after_delay.move_x != 0 or action_after_delay.move_y != 0
