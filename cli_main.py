import argparse

from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration
from aux_utils import convert_image_to_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Acoustic Simulation Toolkit CLI")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3, 4], default=4,
                        help="Simulation mode: 0=AcousticSimulation, 1=TimeReversal, 2=ReverseTimeMigration, 3=TimeReversal+ReverseTimeMigration, 4=Full")
    parser.add_argument("--map", type=str, default="map.png",
                        help="Path to the map image file used to build the velocity model")
    parser.add_argument("--animate", action="store_true",
                        help="Enable animation during simulation")
    return parser.parse_args()


def main():
    args = parse_args()
    # Load velocity map and receiver coordinates from the provided map image
    velocity_map, receptor_z, receptor_x = convert_image_to_matrix(args.map)

    # Simulation configuration dictionary
    simulation_config = {
        "dt": 1e-3,
        "c": velocity_map,
        "dz": 3,
        "dx": 3,
        "grid_size_z": len(velocity_map[:, 0]),
        "grid_size_x": len(velocity_map[0, :]),
        "total_time": 3500,
        "animation_step": 100,
    }

    # Acoustic simulation configuration
    ac_config = {
        "source_z": receptor_z,
        "source_x": receptor_x[0],
        "number_of_receptors": len(receptor_z),
        "receptor_z": receptor_z,
        "receptor_x": receptor_x,
    }

    # Time reversal configuration
    tr_config = {
        "min_time": 0,
        "max_time": simulation_config["total_time"],
        "padding_zeros": 0,
    }

    # Map numeric mode to descriptive mode name
    simulation_modes = {
        0: "AcousticSimulation",
        1: "TimeReversal",
        2: "ReverseTimeMigration",
        3: "TimeReversal + ReverseTimeMigration",
        4: "Full",
    }

    mode_name = simulation_modes.get(args.mode)
    if mode_name is None:
        raise ValueError(f"Invalid simulation mode: {args.mode}")

    # Dispatch to the appropriate simulation(s)
    if mode_name == "AcousticSimulation":
        ac_sim = AcousticSimulation(simulation_config, ac_config)
        ac_sim.run(create_animation=args.animate, plt_kwargs={"cmap": "bwr"})
    elif mode_name == "TimeReversal":
        tr_sim = TimeReversal(simulation_config, tr_config)
        tr_sim.run(create_animation=args.animate, plt_kwargs={"cmap": "bwr"})
    elif mode_name == "ReverseTmeMigration":
        rtm_sim = ReverseTimeMigration(**simulation_config)
        rtm_sim.run(create_animation=args.animate, plt_kwargs={"cmap": "bwr"})
    elif mode_name == "TimeReversal + ReverseTimeMigration":
        tr_sim = TimeReversal(simulation_config, tr_config)
        tr_sim.run(create_animation=args.animate, plt_kwargs={"cmap": "bwr"})
        rtm_sim = ReverseTimeMigration(**simulation_config)
        rtm_sim.run(create_animation=args.animate, plt_kwargs={"cmap": "bwr"})
    elif mode_name == "Full":
        ac_sim = AcousticSimulation(simulation_config, ac_config)
        ac_sim.run(create_animation=args.animate, plt_kwargs={"cmap": "bwr"})
        tr_sim = TimeReversal(simulation_config, tr_config)
        # For the full pipeline we typically do not set a colormap for the time reversal since it feeds into RTM
        tr_sim.run(create_animation=args.animate, plt_kwargs={})
        rtm_sim = ReverseTimeMigration(**simulation_config)
        rtm_sim.run(create_animation=args.animate, plt_kwargs={"cmap": "bwr"})


if __name__ == "__main__":
    main()
