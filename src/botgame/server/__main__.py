import argparse
import time
import random

from botgame.server.world import WorldState, Player, ZoneState, TICK_RATE
from botgame.server.bot_manager import BotManager
from botgame.server.recorder import EpisodeRecorder
from botgame.bots.scripted_utility import ScriptedUtilityBot
from botgame.bots.learned_policy import LearnedPolicyBot # Assuming this is ready

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Run a local bot game server.")
        parser.add_argument("--num_scripted_bots", type=int, default=1, help="Number of scripted bots to include.")
        parser.add_argument("--num_learned_bots", type=int, default=0, help="Number of learned bots to include.")
        parser.add_argument("--episode_duration", type=int, default=60, help="Duration of the episode in seconds.")
        parser.add_argument("--seed", type=int, default=None, help="Random seed for the simulation.")
        parser.add_argument("--record_replay", action="store_true", help="Record replay data for the episode.")
        parser.add_argument("--learned_model_path", type=str, default="artifacts/imitation_policy.pt", help="Path to the learned model.")
        parser.add_argument("--learned_model_type", type=str, default="imitation", choices=["imitation", "ppo"], help="Type of the learned model.")
        args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time())

    print(f"Starting local server with seed: {args.seed}")
    print(f"Scripted bots: {args.num_scripted_bots}, Learned bots: {args.num_learned_bots}")

    # Initialize WorldState
    rng = random.Random(args.seed) # Use system random for initial setup, PRNG for world simulation
    world_rng = PRNG(args.seed)

    players: Dict[str, Player] = {}
    for i in range(args.num_scripted_bots):
        player_id = f"scripted_bot_{i}"
        players[player_id] = Player(id=player_id, is_bot=True, position=[rng.uniform(-20, 20), rng.uniform(-20, 20)],
                                    velocity=[0.0, 0.0], hp=Player.PLAYER_MAX_HP, ammo=Player.PLAYER_MAX_AMMO, cooldowns={})
    for i in range(args.num_learned_bots):
        player_id = f"learned_bot_{i}"
        players[player_id] = Player(id=player_id, is_bot=True, position=[rng.uniform(-20, 20), rng.uniform(-20, 20)],
                                    velocity=[0.0, 0.0], hp=Player.PLAYER_MAX_HP, ammo=Player.PLAYER_MAX_AMMO, cooldowns={})

    # Add a dummy human player for interaction if no bots are present (or always for a baseline)
    if args.num_scripted_bots == 0 and args.num_learned_bots == 0:
        players["human_0"] = Player(id="human_0", is_bot=False, position=[0.0, 0.0], velocity=[0.0, 0.0],
                                     hp=Player.PLAYER_MAX_HP, ammo=Player.PLAYER_MAX_AMMO, cooldowns={})


    initial_zone_state = ZoneState(position=[0.0, 0.0], radius=50.0, is_safe=True)
    world = WorldState(players=players, projectiles={}, items={}, zone_state=initial_zone_state, _rng=world_rng)

    # Initialize BotManager
    bot_manager = BotManager(world)
    for player_id, player in world.players.items():
        if player.is_bot:
            bot_seed = rng.randint(0, 100000)
            if "scripted_bot" in player_id:
                bot_manager.add_bot(player_id, ScriptedUtilityBot(player_id=player_id), seed=bot_seed)
            elif "learned_bot" in player_id:
                try:
                    bot_manager.add_bot(player_id, LearnedPolicyBot(player_id=player_id,
                                                                    model_path=args.learned_model_path,
                                                                    model_type=args.learned_model_type),
                                        seed=bot_seed)
                except Exception as e:
                    print(f"Failed to load learned bot {player_id} policy: {e}. Skipping this bot.")
                    world.players[player_id].hp = 0 # Mark as dead so it's removed
                    continue

    # Initialize Recorder
    recorder = EpisodeRecorder()
    if args.record_replay:
        recorder.start_episode(world)

    # Simulation loop
    dt = 1.0 / TICK_RATE
    total_ticks = int(args.episode_duration * TICK_RATE)

    for tick_num in range(total_ticks):
        if len([p for p in world.players.values() if p.hp > 0 and p.is_bot]) == 0 and 
           len([p for p in world.players.values() if p.hp > 0 and not p.is_bot]) == 0 and 
           (args.num_scripted_bots > 0 or args.num_learned_bots > 0):
            print(f"All bots and human players are dead at tick {tick_num}. Ending episode early.")
            break

        # Store world state before actions and tick for reward calculation
        world_state_before_tick = WorldState(
            players={k: Player(**v.__dict__) for k, v in world.players.items()}, # Deep copy players
            projectiles={k: Projectile(**v.__dict__) for k, v in world.projectiles.items()}, # Deep copy projectiles
            items={k: GameItem(**v.__dict__) for k, v in world.items.items()}, # Deep copy items
            zone_state=ZoneState(**world.zone_state.__dict__), # Deep copy zone
            _rng=PRNG(world._rng._rng.getstate()) # Copy RNG state
        )

        bot_actions_to_record: Dict[str, Action] = {}

        # Bots act
        for player_id, bot_policy in bot_manager.bot_policies.items():
            if player_id in world.players and world.players[player_id].hp > 0:
                obs_for_bot = world.build_observation(player_id)
                action = bot_policy.act(obs_for_bot)
                bot_actions_to_record[player_id] = action
                world.apply_action(player_id, action)

        # Human players (if any) - no actual input, just stand still for now
        for player_id, player in world.players.items():
            if not player.is_bot and player.hp > 0:
                world.apply_action(player_id, Action(0,0,0,0,False, None, False))

        # World simulation tick
        world.tick(dt)

        # Record data for bots
        if args.record_replay:
            for player_id, action in bot_actions_to_record.items():
                is_done = False
                if player_id not in world.players or world.players[player_id].hp <= 0:
                    is_done = True
                recorder.record_step(
                    player_id=player_id,
                    observation=world_state_before_tick.build_observation(player_id), # Obs before action
                    action=action,
                    world_state_before=world_state_before_tick,
                    world_state_after=world,
                    is_done=is_done
                )

        # Simple console output
        if tick_num % (TICK_RATE * 5) == 0: # Every 5 seconds
            print(f"--- Tick {world.tick_id} (Time: {world.time:.2f}s) ---")
            for p_id, player in world.players.items():
                if player.hp > 0:
                    print(f"  Player {p_id} (HP: {player.hp:.1f}, Ammo: {player.ammo}, Pos: [{player.position[0]:.1f}, {player.position[1]:.1f}])")
                else:
                    print(f"  Player {p_id} (Dead)")

    print("Episode finished.")
    if args.record_replay:
        recorder.end_episode(episode_id=f"match_seed{args.seed}_duration{args.episode_duration}")
        print(f"Replay data recorded to {recorder.output_dir}")

if __name__ == "__main__":
    main()
