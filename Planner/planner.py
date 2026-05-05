import math
import time
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely import Point, LineString
from .planner_utils import *
from .observation import *
from GameFormer.predictor import GameFormer
from GameFormer.data_utils import create_map_raster, create_ego_raster, create_agents_raster
from .state_lattice_path_planner import LatticePlanner

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


class Planner(AbstractPlanner):
    def __init__(self, model_path, device=None, debug=False, debug_dir=None, debug_max_plots=50):
        self._max_path_length = MAX_LEN # [m]
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._target_speed = 13.0 # [m/s]
        self._N_points = int(T/DT)
        self._model_path = model_path
        self._debug = debug
        self._debug_dir = debug_dir
        self._debug_max_plots = debug_max_plots
        self._debug_plot_count = 0

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self._device = device
    
    def name(self) -> str:
        return "GameFormer Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TrajectoryPlanner()
        self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length)

    def _initialize_model(self):
        # The parameters of the model should be the same as the one used in training
        self._model = GameFormer(encoder_layers=3, decoder_levels=2)
        
        # Load trained model
        self._model.load_state_dict(torch.load(self._model_path, map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        
    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]
    
    def _get_reference_path(self, ego_state, traffic_light_data, observation):
        # Get starting block
        starting_block = None
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
            
        # In case the ego vehicle is not on the route, return None
        if closest_distance > 5:
            return None

        # Get reference path, handle exception
        try:
            ref_path = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        except:
            ref_path = None

        if ref_path is None:
            return None

        # Annotate red light to occupancy
        occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                conn_path = lane_conn.baseline_path.discrete_path
                conn_path = np.array([[p.x, p.y] for p in conn_path])
                red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
        if len(ref_path) < MAX_LEN * 10:
            ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)
        
        return ref_path.astype(np.float32)
    
    def get_candidate_routes_bfs(self, ego_state, max_routes=5, points_per_route=50, search_distance=150.0):
         import numpy as np
         from nuplan.common.actor_state.state_representation import Point2D
         from nuplan.common.maps.maps_datatypes import SemanticMapLayer
         
         ego_x = ego_state.rear_axle.x
         ego_y = ego_state.rear_axle.y
         ego_heading = ego_state.rear_axle.heading
         ego_point = Point2D(ego_x, ego_y)
         
         # 1. Başlangıç Şeritlerini Bul 
         layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
         current_map_objs = self._map_api.get_proximal_map_objects(ego_point, 5.0, layers)
         
         c_lat_candidates = np.zeros((max_routes, points_per_route, 3))
         valid_start_lanes = []
         
         if current_map_objs:
             for layer in layers:
                 if layer in current_map_objs and current_map_objs[layer]:
                     for lane in current_map_objs[layer]:
                         lane_pts = lane.baseline_path.discrete_path
                         if len(lane_pts) > 0:
                             mid_pt = lane_pts[len(lane_pts)//2]
                             heading_diff = mid_pt.heading - ego_heading
                             if np.cos(heading_diff) > 0.5: 
                                 valid_start_lanes.append(lane)
                                 
         if not valid_start_lanes:
             return c_lat_candidates
             
         def dist_to_ego(lane):
             pts = lane.baseline_path.discrete_path
             return min((pt.x - ego_x)**2 + (pt.y - ego_y)**2 for pt in pts)
             
         valid_start_lanes.sort(key=dist_to_ego)
         queue = [(lane, [lane], 0.0) for lane in valid_start_lanes]
         candidate_paths = []
         
         while queue and len(candidate_paths) < max_routes:
             current_lane, path_lanes, path_length = queue.pop(0)
             
             lane_pts = current_lane.baseline_path.discrete_path
             if len(lane_pts) > 1:
                 lane_len = np.hypot(lane_pts[-1].x - lane_pts[0].x, lane_pts[-1].y - lane_pts[0].y)
             else:
                 lane_len = 0.0
                 
             new_length = path_length + lane_len
             
             if new_length >= search_distance:
                 candidate_paths.append(path_lanes)
                 continue
                 
             # SADECE İLERİYE BAĞLANAN (OUTGOING) ŞERİTLERİ ALIYORUZ!
             # (Adjacent edges saçmalığını tamamen sildim)
             next_possible_lanes = []
             if current_lane.outgoing_edges:
                 next_possible_lanes.extend(list(current_lane.outgoing_edges))
             
             if not next_possible_lanes:
                 candidate_paths.append(path_lanes)
             else:
                 for next_lane in next_possible_lanes:
                     if next_lane is not None:
                         # Geriye dönmeyi / kendi üstüne katlanmayı engelle
                         if next_lane.id not in [l.id for l in path_lanes]:
                             queue.append((next_lane, path_lanes + [next_lane], new_length))
                         
         # 3. GLOBAL KOORDİNATLARI EGO-CENTRIC (RELATİF) KOORDİNATLARA ÇEVİR VE KIRP
         c, s = np.cos(-ego_heading), np.sin(-ego_heading)
         
         for i, lane_path in enumerate(candidate_paths):
             if i >= max_routes:
                 break
                 
             full_centerline = []
             for lane in lane_path:
                 for point in lane.baseline_path.discrete_path:
                     dx = point.x - ego_x
                     dy = point.y - ego_y
                     
                     rel_x = dx * c - dy * s
                     rel_y = dx * s + dy * c
                     rel_yaw = point.heading - ego_heading
                     
                     if rel_x > -2.0:
                         full_centerline.append([rel_x, rel_y, rel_yaw])
                     
             full_centerline = np.array(full_centerline)
             
             if len(full_centerline) > 1:
                 orig_indices = np.linspace(0, 1, len(full_centerline))
                 target_indices = np.linspace(0, 1, points_per_route)
                 
                 ref_x = np.interp(target_indices, orig_indices, full_centerline[:, 0])
                 ref_y = np.interp(target_indices, orig_indices, full_centerline[:, 1])
                 ref_yaw = np.interp(target_indices, orig_indices, full_centerline[:, 2])
                 
                 c_lat_candidates[i] = np.stack([ref_x, ref_y, ref_yaw], axis=-1)
         return c_lat_candidates

    def _get_prediction(self, features):
        predictions, ego_plan, neural_plan = self._model(features)
        K = len(predictions) // 2 - 1
        final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
        final_scores = predictions[f'level_{K}_scores']
        ego_current = features['ego_agent_past'][:, -1]
        neighbors_current = features['neighbor_agents_past'][:, :, -1]

        return ego_plan, neural_plan, final_predictions, final_scores, ego_current, neighbors_current

    def _save_debug_plot(self, features, ref_path, best_c_lat, neural_plan, ego_plan, final_path, iteration=0):
        if not self._debug:
            return
        if self._debug_plot_count >= self._debug_max_plots:
            return

        out_dir = self._debug_dir or "testing_log/debug_plots"
        os.makedirs(out_dir, exist_ok=True)

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)

        # --- HARİTA VE GEÇMİŞ AJAN ÇİZİMLERİ (GERİ EKLENDİ) ---
        map_lanes = features['map_lanes'][0].detach().cpu().numpy()
        map_crosswalks = features['map_crosswalks'][0].detach().cpu().numpy()
        route_lanes = features['route_lanes'][0].detach().cpu().numpy()
        ego_past = features['ego_agent_past'][0].detach().cpu().numpy()
        neighbors_past = features['neighbor_agents_past'][0].detach().cpu().numpy()

        create_map_raster(map_lanes, map_crosswalks, route_lanes)
        create_ego_raster(ego_past[-1])
        create_agents_raster(neighbors_past[:, -1])
        
        ax.plot(ego_past[:, 0], ego_past[:, 1], color='#00a8e8', linewidth=2.0, alpha=0.95, zorder=4, label='ego_past')
        for i in range(neighbors_past.shape[0]):
            if neighbors_past[i, -1, 0] != 0:
                ax.plot(neighbors_past[i, :, 0], neighbors_past[i, :, 1], color='m', linewidth=1.0, alpha=0.6, zorder=3)

        # --- FİLTRELENMİŞ TEMEL YÖRÜNGELER ---
        if ref_path is not None:
            ax.plot(ref_path[:, 0], ref_path[:, 1], linestyle='--', linewidth=2.0, color='k', zorder=6, label='ref_path')

        if best_c_lat is not None and np.any(best_c_lat):
            ax.plot(best_c_lat[:, 0], best_c_lat[:, 1], linestyle=':', linewidth=4.0, color='gray', zorder=5, label='best_c_lat (Mode)')

        if neural_plan is not None:
            neural_plan = np.asarray(neural_plan)
            ax.plot(neural_plan[:, 0], neural_plan[:, 1], linewidth=2.8, color='tab:orange', zorder=8, label='neural_plan')
            ax.scatter(neural_plan[-1, 0], neural_plan[-1, 1], color='tab:orange', s=30, zorder=9, marker='*')

        if ego_plan is not None:
            ego_plan = np.asarray(ego_plan)
            ax.plot(ego_plan[:, 0], ego_plan[:, 1], linewidth=2.8, color='tab:blue', zorder=7, label='ego_plan')
            ax.scatter(ego_plan[-1, 0], ego_plan[-1, 1], color='tab:blue', s=28, zorder=8)

        if final_path is not None:
            final_path = np.asarray(final_path)
            ax.plot(final_path[:, 0], final_path[:, 1], linewidth=3.2, color='tab:red', zorder=10, label='final_path')
            ax.scatter(final_path[-1, 0], final_path[-1, 1], color='tab:red', s=35, zorder=11, marker='D')

        ax.scatter([0.0], [0.0], marker='x', s=60, color='black', label='ego_origin', zorder=12)
        
        ax.set_title(f'Debug Scenario Iter {iteration}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        legend_handles = [
            Line2D([0], [0], color='c', lw=3, label='lanes'),
            Line2D([0], [0], color='b', lw=4, label='crosswalks'),
            Line2D([0], [0], color='g', lw=4, label='route_lanes'),
            Line2D([0], [0], color='#00a8e8', lw=2, label='ego_past'),
            Line2D([0], [0], color='k', lw=2, linestyle='--', label='ref_path'),
            Line2D([0], [0], color='gray', lw=4, linestyle=':', label='best_c_lat (Mode)'),
            Line2D([0], [0], color='tab:orange', lw=3, label='neural_plan'),
            Line2D([0], [0], color='tab:blue', lw=3, label='ego_plan'),
            Line2D([0], [0], color='tab:red', lw=3, label='final_path'),
        ]
        ax.legend(handles=legend_handles, loc='best')

        file_name = os.path.join(out_dir, f'debug_iter_{iteration:04d}.png')
        fig.savefig(file_name, dpi=120, bbox_inches='tight')
        plt.close(fig)
        self._debug_plot_count += 1
    
    def _plan(self, ego_state, history, traffic_light_data, observation, iteration=None):
        # Construct input features
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # 2. YENİ: Aday rotaları bul ve tensör olarak features içine ekle
        c_lat_candidates = self.get_candidate_routes_bfs(ego_state)
        features['c_lat_candidates'] = torch.tensor(c_lat_candidates, dtype=torch.float32, device=self._device).unsqueeze(0)

        # Get reference path
        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        # Infer prediction model
        with torch.no_grad():
            ego_plan, neural_plan ,predictions, scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features)
            

        # Trajectory refinement
        with torch.no_grad():
            final_plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed, 
                                                 predictions, neural_plan, scores, ref_path, observation)



        # 6. Debug Çizimi
        if self._debug:
            self._save_debug_plot(
                features=features, # HARİTA VERİSİ EKLENDİ
                ref_path=ref_path,
                best_c_lat=None,
                neural_plan=neural_plan[0].detach().cpu().numpy(),
                ego_plan=ego_plan[0].detach().cpu().numpy() if ego_plan is not None else None,
                final_path=final_plan,
                iteration=0 if iteration is None else iteration,
            )

        # Çıktıları numpy'a çevir (Eski kodlar)
        
        neural_plan = neural_plan[0].cpu().numpy()
        ego_plan = ego_plan[0].cpu().numpy() 
        print(f"Neural Plan shape: {neural_plan.shape}, Ego Plan shape: {ego_plan.shape}")
        states = transform_predictions_to_states(ego_plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)

        return trajectory
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        trajectory = self._plan(ego_state, history, traffic_light_data, observation, iteration=iteration)
        # print(f'Iteration {iteration}: {time.time() - s:.3f} s')  # Commented out: verbose iteration logging

        return trajectory
