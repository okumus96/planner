import time
import yaml
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore") 

from tqdm import tqdm
from Planner.plannerv2 import Planner
from common_utils import *

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile


def build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir):
    """
    Builds the main experiment folder for simulation.
    :return: The main experiment folder path.
    """
    print('Building experiment folders...')

    exp_folder = pathlib.Path(output_dir)
    print(f'\nFolder where all results are stored: {exp_folder}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Build nuboard event file.
    nuboard_filename = exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
    nuboard_file = NuBoardFile(
        simulation_main_path=str(exp_folder),
        simulation_folder=simulation_dir,
        metric_main_path=str(exp_folder),
        metric_folder=metric_dir,
        aggregator_metric_folder=aggregator_metric_dir,
    )

    metric_main_path = exp_folder / metric_dir
    metric_main_path.mkdir(parents=True, exist_ok=True)

    nuboard_file.save_nuboard_file(nuboard_filename)
    print('Building experiment folders...DONE!')

    return exp_folder.name


def build_simulation(experiment, planner, scenarios, output_dir, simulation_dir, metric_dir):
    runner_reports = []
    print(f'Building simulations from {len(scenarios)} scenarios...')

    metric_engine = build_metrics_engine(experiment, output_dir, metric_dir)
    print('Building metric engines...DONE\n')

    # Iterate through scenarios
    for scenario in tqdm(scenarios, desc='Running simulation'):
        # M1 oracle modu icin: planner uzman gelecege senaryo uzerinden erisir.
        planner._oracle_scenario = scenario
        tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                            r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                            jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                            stopping_proportional_gain=0.5, stopping_velocity=0.2)
        motion_model = KinematicBicycleModel(get_pacifica_parameters())

        # Ego Controller and Perception
        if experiment == 'open_loop_boxes':
            ego_controller = LogPlaybackController(scenario) 
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_nonreactive_agents': 
            ego_controller = TwoStageController(scenario, tracker, motion_model)
            observations = TracksObservation(scenario)
        else:      
            ego_controller = TwoStageController(scenario, tracker, motion_model)
            observations = IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                     accel_max=1.0, decel_max=2.0, scenario=scenario,
                                     open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", 
                                                                 "TRAFFIC_CONE", "GENERIC_OBJECT"])

        # Simulation Manager
        simulation_time_controller = StepSimulationTimeController(scenario)

        # Stateful callbacks
        metric_callback = MetricCallback(metric_engine=metric_engine)
        sim_log_callback = SimulationLogCallback(output_dir, simulation_dir, "msgpack")

        # Construct simulation and manager
        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )

        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([metric_callback, sim_log_callback])
        )

        # Begin simulation
        simulation_runner = SimulationRunner(simulation, planner)
        report = simulation_runner.run()
        runner_reports.append(report)
    
    # save reports
    save_runner_reports(runner_reports, output_dir, 'runner_reports')

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0

    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            print("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = len(scenarios) - number_of_successful
    print(f"Number of successful simulations: {number_of_successful}")
    print(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        print(f"Failed simulations [log, token]:\n{failed_simulations}")
    
    print('Finished running simulations!')

    return runner_reports


def build_nuboard(scenario_builder, simulation_path, port_number=5006):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=port_number,
    )

    nuboard.run()


def main(args):
    # parameters
    experiment_name = args.experiment_name
    job_name = 'gameformer_planner'
    experiment_time = datetime.datetime.now()
    experiment = f"{experiment_name}/{job_name}/{experiment_time}"  
    output_dir = f"testing_log/{experiment}"
    simulation_dir = "simulation"
    metric_dir = "metrics"
    aggregator_metric_dir = "aggregator_metric"

    # initialize planner
    if args.causal_path and args.deploy == 'pure':
        from Planner.causal_planner_pure import CausalPurePlanner
        planner = CausalPurePlanner(
            backbone_path=args.model_path, causal_path=args.causal_path,
            num_neighbors=args.num_neighbors, graph_layers=args.graph_layers,
            modes=args.modes, plan_source=args.plan_source, nbr_enrich=args.nbr_enrich, ego_residual=args.ego_residual,
            gate_channels=args.gate_channels, typed_kv=args.typed_kv, joint_softmax=args.joint_softmax,
            channel_evidence=args.channel_evidence, gate_trust=args.gate_trust,
            dod_meta=args.dod_meta, lon_merge=args.lon_merge, device=args.device,
        )
        print(f"[PURE CAUSAL] backbone={args.model_path}  causal={args.causal_path}  (refiner YOK)  "
              f"plan_source={args.plan_source}")
    elif args.causal_path:  # deploy == 'refiner'
        from Planner.causal_refiner_planner import CausalRefinerPlanner
        planner = CausalRefinerPlanner(
            backbone_path=args.model_path, causal_path=args.causal_path,
            num_neighbors=args.num_neighbors, graph_layers=args.graph_layers,
            modes=args.modes,
            use_causal=(not args.baseline), remove=args.remove, remove_k=args.remove_k,
            plan_source=args.plan_source, nbr_enrich=args.nbr_enrich, ego_residual=args.ego_residual,
            gate_channels=args.gate_channels, typed_kv=args.typed_kv, joint_softmax=args.joint_softmax,
            channel_evidence=args.channel_evidence, gate_trust=args.gate_trust,
            dod_meta=args.dod_meta, lon_merge=args.lon_merge,
            uniform_mask=args.uniform_mask, device=args.device,
        )
        print(f"[CAUSAL+REFINER] causal={args.causal_path}  "
              f"plan={'GameFormer' if args.baseline else 'CausalPlanner'}  remove={args.remove}x{args.remove_k}  "
              f"plan_source={args.plan_source}  channels(gate={args.gate_channels},typed={args.typed_kv},"
              f"evid={args.channel_evidence},trust={args.gate_trust})")
    else:
        planner = Planner(
            args.model_path,
            args.device,
            debug=args.debug,
            debug_dir=f"{output_dir}/debug_plots",
            debug_max_plots=args.debug_max_plots,
            oracle_mode=args.oracle_mode,
        )
    if args.oracle_mode:
        print("[M1] ORACLE MODE: ModeSelector bypass, mod uzman gelecekten secilecek.")

    # initialize main aggregator
    metric_aggregators = build_metrics_aggregators(experiment_name, output_dir, aggregator_metric_dir)
    metric_save_path = f"{output_dir}/{metric_dir}"
    metric_aggregator_callback = MetricAggregatorCallback(metric_save_path, metric_aggregators)
    metric_file_callback = MetricFileCallback(metric_file_output_path=f"{output_dir}/{metric_dir}",
                                              scenario_metric_paths=[f"{output_dir}/{metric_dir}"],
                                              delete_scenario_metric_files=True)
    metric_summary_callback = MetricSummaryCallback(metric_save_path=f"{output_dir}/{metric_dir}",
                                                    metric_aggregator_save_path=f"{output_dir}/{aggregator_metric_dir}",
                                                    summary_output_path=f"{output_dir}/summary",
                                                    num_bins=20, pdf_file_name='summary.pdf')
    main_callbacks = MultiMainCallback([metric_file_callback, metric_aggregator_callback, metric_summary_callback])
    main_callbacks.on_run_simulation_start()

    # build simulation folder
    build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir)

    # build scenarios
    print('Extracting scenarios...')
    data_root = args.data_path
    map_root = args.map_path
    sensor_root = None
    db_files = None
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(data_root, map_root, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
    
    # Load filter parameters from config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        scenarios_per_type = config.get('num_scenarios_per_type')
        total_scenarios = config.get('limit_total_scenarios')
        shuffle_scenarios = config.get('shuffle', False)
        scenario_types = config.get('scenario_types')
        scenario_tokens = config.get('scenario_tokens')
        log_names = config.get('log_names')
        map_names = config.get('map_names')
        timestamp_threshold_s = config.get('timestamp_threshold_s')
        ego_displacement_minimum_m = config.get('ego_displacement_minimum_m')
        expand_scenarios = config.get('expand_scenarios')
        remove_invalid_goals = config.get('remove_invalid_goals')
        ego_start_speed_threshold = config.get('ego_start_speed_threshold')
        ego_stop_speed_threshold = config.get('ego_stop_speed_threshold')
        speed_noise_tolerance = config.get('speed_noise_tolerance')
        print(f"Loaded config from: {args.config}")
    else:
        scenarios_per_type = args.scenarios_per_type
        total_scenarios = args.total_scenarios
        shuffle_scenarios = args.shuffle_scenarios
        scenario_types = None
        scenario_tokens = None
        log_names = None
        map_names = None
        timestamp_threshold_s = None
        ego_displacement_minimum_m = None
        expand_scenarios = None
        remove_invalid_goals = None
        ego_start_speed_threshold = None
        ego_stop_speed_threshold = None
        speed_noise_tolerance = None
    
    # Get filter parameters with defaults
    filter_params_default = list(get_filter_parameters(scenarios_per_type, total_scenarios, shuffle_scenarios))
    
    # Override with config values where provided
    if scenario_types is not None:
        filter_params_default[0] = scenario_types
    if scenario_tokens is not None:
        filter_params_default[1] = scenario_tokens
    if log_names is not None:
        filter_params_default[2] = log_names
    if map_names is not None:
        filter_params_default[3] = map_names
    if timestamp_threshold_s is not None:
        filter_params_default[6] = timestamp_threshold_s
    if ego_displacement_minimum_m is not None:
        filter_params_default[7] = ego_displacement_minimum_m
    if expand_scenarios is not None:
        filter_params_default[8] = expand_scenarios
    if remove_invalid_goals is not None:
        filter_params_default[9] = remove_invalid_goals
    if ego_start_speed_threshold is not None:
        filter_params_default[11] = ego_start_speed_threshold
    if ego_stop_speed_threshold is not None:
        filter_params_default[12] = ego_stop_speed_threshold
    if speed_noise_tolerance is not None:
        filter_params_default[13] = speed_noise_tolerance
    
    scenario_filter = ScenarioFilter(*filter_params_default)
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    del worker, scenario_filter, scenario_mapping
    
    # Check if scenarios were found
    if len(scenarios) == 0:
        print("\n" + "="*80)
        print("ERROR: No scenarios found with the current filtering criteria!")
        print("="*80)
        if args.config:
            print(f"\nConfig file: {args.config}")
            print("Please check your config file's filtering parameters:")
            print("  - scenario_tokens (if using specific tokens)")
            print("  - scenario_types")
            print("  - log_names")
            print("  - map_names")
            print("  - timestamp_threshold_s")
            print("  - ego_displacement_minimum_m")
        else:
            print(f"\nCurrent filters:")
            print(f"  - scenarios_per_type: {args.scenarios_per_type}")
            print(f"  - total_scenarios: {args.total_scenarios}")
        print("="*80 + "\n")
        return

    # begin testing
    build_simulation(experiment_name, planner, scenarios, output_dir, simulation_dir, metric_dir)
    main_callbacks.on_run_simulation_end()
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']

    # show metrics and scenarios
    if args.no_nuboard:
        print(f"[no_nuboard] Skipping nuBoard. Results: {output_dir}")
    else:
        build_nuboard(builder, simulation_file, port_number=args.nuboard_port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NuPlan test')
    parser.add_argument('--experiment_name', choices=['open_loop_boxes', 'closed_loop_nonreactive_agents', 
                                                      'closed_loop_reactive_agents'], help='experiment name')
    parser.add_argument('--config', type=str, help='path to test scenario filter config YAML (e.g., test14-random.yaml)')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--map_path', type=str, help='path to nuplan maps')
    parser.add_argument('--model_path', type=str, help='path to model (frozen GameFormer backbone)')
    parser.add_argument('--joint_softmax', type=int, default=0,
                        help='1 = ajan+harita ORTAK softmax paydasi. Checkpoint hangi modda '
                             'egitildiyse o verilmeli (agirliksiz anahtar, strict=False uyarmaz).')
    parser.add_argument('--ego_residual', type=int, default=1,
                        help='checkpoint hangi degerle EGITILDIYSE o verilmeli (0 = h_ego residual yok)')
    parser.add_argument('--causal_path', type=str, default=None,
                        help='CausalPlanner ckpt -> PURE causal planner (refiner YOK). Verilirse model_path=backbone.')
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--graph_layers', type=int, default=3)
    parser.add_argument('--nbr_enrich', type=int, default=0)
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--deploy', type=str, default='refiner', choices=['pure', 'refiner'],
                        help='causal cikti: pure (refiner yok) | refiner (causal neural_plan -> refiner)')
    parser.add_argument('--baseline', action='store_true',
                        help='neural_plan icin CausalPlanner yerine GameFormer kullan (maske-yok baseline).')
    parser.add_argument('--remove', type=str, default='none', choices=['none', 'high', 'low', 'random', 'cfd_high'],
                        help='RemoveNonCausal-via-CLS: her frame M_cas gore ajan cikar. '
                             'high=en causal (CLS dusmeli), low=en az causal (degismemeli), random=kontrol.')
    parser.add_argument('--plan_source', type=str, default='cas', choices=['cas', 'cfd'],
                        help="plan hangi dalin f'inden uretilsin: 'cas' (varsayilan, ANA plan) vs "
                             "'cfd' (confounding graph'tan uret, HICBIR ajan silinmez, remove'dan bagimsiz — "
                             "'confounding graph gercekten davranis-belirleyici mi?' testi).")
    parser.add_argument('--remove_k', type=int, default=1, help='kac ajan cikarilacak (high/low top-k). default 1.')
    parser.add_argument('--gate_channels', type=int, default=0,
                        help='ckpt hangi degerle egitildiyse o: predicate kanali yanmayan girdi causal '
                             'softmax\'a giremez. Deployment\'ta kanallar on-the-fly hesaplanir '
                             '(GF top-1 future + lattice ref path). Sadece --deploy refiner.')
    parser.add_argument('--typed_kv', type=int, default=0,
                        help='ckpt hangi degerle egitildiyse o: (ajan, kanal) tipli K/V + tipli causal softmax.')
    parser.add_argument('--channel_evidence', type=int, default=0,
                        help='ckpt hangi degerle egitildiyse o: kanal evidence vektorleri edge feature\'a eklenir.')
    parser.add_argument('--gate_trust', type=str, default='all', choices=['all', 'reliable'],
                        help='gate karari hangi kanallara guvensin (reliable: zayif-IoU kanallar sayilmaz).')
    parser.add_argument('--dod_meta', type=int, default=0,
                        help='ckpt hangi degerle egitildiyse o: factored (lon x lat) meta-aksiyon DOD\'u (H).')
    parser.add_argument('--lon_merge', type=int, default=0,
                        help='ckpt lon_merge=1 ile egitildiyse o (6 lon sinifi).')
    parser.add_argument('--uniform_mask', type=int, default=0,
                        help='RULES-ONLY baseline: kural girdiyi secer, AGIRLIK ogrenilmez -- '
                             'gate\'ten gecen girdiler uzerinde uniform. Ogrenilmis tahsisin '
                             'katkisini olcer (egitim gerekmez, inference-time).')
    parser.add_argument('--device', type=str, default='cuda', help='device to run model on')
    parser.add_argument('--debug', action='store_true', help='save per-iteration debug trajectory plots')
    parser.add_argument('--debug_max_plots', type=int, default=200, help='maximum number of debug plots to save')
    parser.add_argument('--oracle_mode', action='store_true',
                        help='M1: bypass ModeSelector, pick mode from expert (log) future — mode-interface upper bound')
    parser.add_argument('--no_nuboard', action='store_true', help='do not launch nuBoard after simulation (batch runs)')
    parser.add_argument('--nuboard_port', type=int, default=5006,
                        help='nuBoard http port (default nuplan devkit default 5006) — verilmezse PARALEL '
                             'kosular hepsi ayni portta cakisir; her koşuya FARKLI port ver')
    parser.add_argument('--scenarios_per_type', type=int, default=10, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', default=None, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=False, help='shuffle scenarios')
    args = parser.parse_args()

    main(args)
