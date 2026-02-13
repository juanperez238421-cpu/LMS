import argparse
from botgame.server.__main__ import main as server_main

def main():
    parser = argparse.ArgumentParser(description="Run a bot game match with configurable parameters.")
    parser.add_argument("--num_scripted_bots", type=int, default=1, help="Number of scripted bots to include.")
    parser.add_argument("--num_learned_bots", type=int, default=0, help="Number of learned bots to include.")
    parser.add_argument("--episode_duration", type=int, default=60, help="Duration of the episode in seconds.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the simulation.")
    parser.add_argument("--record_replay", action="store_true", help="Record replay data for the episode.")
    parser.add_argument("--learned_model_path", type=str, default="artifacts/imitation_policy.pt", help="Path to the learned model.")
    parser.add_argument("--learned_model_type", type=str, default="imitation", choices=["imitation", "ppo"], help="Type of the learned model.")
    args = parser.parse_args()

    # Pass parsed arguments to the server's main function
    # This is a bit indirect, normally you'd call server_main with the arguments directly,
    # or refactor server_main to accept an args object.
    # For now, we'll mimic the command line behavior.
    # A cleaner approach would be to have server_main accept **kwargs or an argparse Namespace object.

    # This works because argparse.parse_args() by default uses sys.argv,
    # so calling it again in server_main will re-parse the same arguments if we don't
    # explicitly pass them. To avoid re-parsing, we should call server_main with an args object.
    
    # For now, let's assume `server_main` will be updated to accept an `args` namespace.
    # To make it runnable without modifying server_main yet, we can temporarily recreate sys.argv
    # or more cleanly pass the args object if server_main was designed for it.

    server_main(args)

if __name__ == "__main__":
    main()
