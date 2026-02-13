import pytest
from botgame.server.world import WorldState, Player, ZoneState, TICK_RATE
from botgame.common.rng import PRNG

def test_world_tick_advances_time():
    seed = 123
    rng = PRNG(seed)
    initial_players = {"p1": Player(id="p1", is_bot=False, position=[0,0], velocity=[0,0], hp=100, ammo=30, cooldowns={})}
    initial_zone = ZoneState(position=[0,0], radius=50, is_safe=True)
    world = WorldState(players=initial_players, projectiles={}, items={}, zone_state=initial_zone, _rng=rng)

    initial_time = world.time
    dt = 1.0 / TICK_RATE
    world.tick(dt)

    assert world.time == pytest.approx(initial_time + dt)
    assert world.tick_id == 1

def test_player_movement():
    seed = 123
    rng = PRNG(seed)
    p1 = Player(id="p1", is_bot=False, position=[0,0], velocity=[1.0, 0.0], hp=100, ammo=30, cooldowns={})
    initial_players = {"p1": p1}
    initial_zone = ZoneState(position=[0,0], radius=50, is_safe=True)
    world = WorldState(players=initial_players, projectiles={}, items={}, zone_state=initial_zone, _rng=rng)

    dt = 1.0
    world.tick(dt) # Tick for 1 second

    assert world.players["p1"].position[0] == pytest.approx(1.0 * dt)
    assert world.players["p1"].position[1] == pytest.approx(0.0 * dt)

def test_projectile_movement_and_collision():
    seed = 123
    rng = PRNG(seed)
    # Player 1 shoots at Player 2 who is stationary
    p1 = Player(id="p1", is_bot=False, position=[0,0], velocity=[0,0], hp=100, ammo=30, cooldowns={})
    p2 = Player(id="p2", is_bot=False, position=[5,0], velocity=[0,0], hp=100, ammo=30, cooldowns={})
    
    initial_players = {"p1": p1, "p2": p2}
    initial_zone = ZoneState(position=[0,0], radius=50, is_safe=True)
    world = WorldState(players=initial_players, projectiles={}, items={}, zone_state=initial_zone, _rng=rng)

    # Manually create a projectile
    from botgame.server.world import Projectile, PROJECTILE_SPEED, PROJECTILE_DAMAGE
    proj_id = "test_proj_0"
    projectile = Projectile(
        id=proj_id,
        owner_id="p1",
        position=[0.1, 0.0], # Start just in front of p1
        velocity=[PROJECTILE_SPEED, 0.0],
        damage=PROJECTILE_DAMAGE,
        creation_time=world.time
    )
    world.projectiles = {proj_id: projectile}

    dt = 0.1 # Small dt for precise collision

    # Tick until projectile should hit p2
    ticks_to_hit = int(5 / PROJECTILE_SPEED / dt) + 1 # +1 to ensure it passes the distance check
    
    initial_p2_hp = world.players["p2"].hp
    
    for _ in range(ticks_to_hit):
        world.tick(dt)
        if world.players["p2"].hp < initial_p2_hp:
            break # Collision detected

    assert world.players["p2"].hp == pytest.approx(initial_p2_hp - PROJECTILE_DAMAGE)
    assert proj_id not in world.projectiles # Projectile should be removed after hit

def test_player_dies_outside_zone():
    seed = 123
    rng = PRNG(seed)
    p1 = Player(id="p1", is_bot=False, position=[100,100], velocity=[0,0], hp=10, ammo=30, cooldowns={}) # Outside zone
    initial_players = {"p1": p1}
    initial_zone = ZoneState(position=[0,0], radius=5.0, is_safe=True) # Small zone
    world = WorldState(players=initial_players, projectiles={}, items={}, zone_state=initial_zone, _rng=rng)

    dt = 1.0 # 1 second per tick for simplicity
    
    initial_hp = world.players["p1"].hp

    # Tick until HP drops below 0
    ticks_to_die = int(initial_hp / (1.0 * dt)) + 1 # Damage is 1.0 * dt per second
    
    for _ in range(ticks_to_die):
        world.tick(dt)
        if "p1" not in world.players:
            break # Player removed

    assert "p1" not in world.players # Player should be removed
