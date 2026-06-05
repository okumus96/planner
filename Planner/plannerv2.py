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
from GameFormer.ar_wrapper import *
from GameFormer.train_utils import sort_candidates_by_lateral
from GameFormer.relevance_graph import (SceneRelevanceGraph, plot_scene_graph, annotate_map_node_ids,
                                        plot_bev_relevance, build_relevance_record, draw_relevance)
import pickle
from Planner.cubic_spline import calc_spline_course


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
        self._debug_candidates_plot_count = 0
        # SceneRelevanceGraph hard top-k secimi (None = soft bias). Inference'ta None disinda
        # bir k verilirse en onemli k dugum disindakiler maskelenir.
        self._graph_hard_topk = None
        self._prev_importance = None  # son adimin grafik onem ciktisi (viz icin)
        # Debug onem gorsellestirme: NORMALIZE onem (imp/max) bu esigin ustundekiler vurgulanir.
        self._relevance_threshold = 0.65
        self._relevance_records = []  # senaryo boyunca per-adim onem verisi (--debug ile kaydedilir)

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

        # --- YENİ: SENARYO BAZLI KLASÖRLEME VE SAYAÇ SIFIRLAMA ---
        if not hasattr(self, '_scenario_idx'):
            self._scenario_idx = 0
        self._scenario_idx += 1
        
        base_dir = self._debug_dir or "testing_log/debug_plots"
        self._current_scenario_dir = os.path.join(base_dir, f"scenario_{self._scenario_idx:02d}")
        
        # Her yeni senaryoda sayaçları sıfırla ki "--debug_max_plots 200" limiti HER senaryo için baştan başlasın
        self._debug_plot_count = 0
        self._debug_candidates_plot_count = 0
        self._relevance_records = []  # yeni senaryo -> onem kaydini sifirla

    def _initialize_model(self):
        # 1. OMURGA (BACKBONE): Orijinal eğitilmiş GameFormer'ı yükle ve dondur
        self.backbone = GameFormer(encoder_layers=3, decoder_levels=2)
        
        # DİKKAT: Buraya orijinal (dondurduğunuz) GameFormer modelinin tam yolunu yazın
        backbone_path = "/home/lt-hta-ai4/GameFormer-Planner/training_log/normal/model_epoch_19_valADE_1.6487.pth" 
        self.backbone.load_state_dict(torch.load(backbone_path, map_location=self._device))
        self.backbone.to(self._device)
        self.backbone.eval()
        
        # 2. PLANLAYICI BAŞLIK (HEAD): Yeni eğittiğiniz AR_Planner'ı yükle
        # self._model_path parametresi, test komutundan gelen yeni eğitilmiş model yoludur
        self.planner_head = ModeSelector()
        self.planner_head.load_state_dict(torch.load(self._model_path, map_location=self._device))
        self.planner_head.to(self._device)
        self.planner_head.eval()

        # 3. SAHNE ONEM GRAFIGI (opsiyonel): mode_selector ile birlikte egitildiyse, ayni
        # klasordeki 'relevance_graph_*.pth' checkpoint'i otomatik yuklenir. Bulunamazsa
        # mode_selector grafiksiz calisir (eski davranis).
        self.relevance_graph = None
        graph_path = self._model_path.replace("mode_selector", "relevance_graph")
        if graph_path != self._model_path and os.path.exists(graph_path):
            self.relevance_graph = SceneRelevanceGraph()
            self.relevance_graph.load_state_dict(torch.load(graph_path, map_location=self._device))
            self.relevance_graph.to(self._device)
            self.relevance_graph.eval()
            print(f"[Planner] SceneRelevanceGraph yuklendi: {graph_path}")
        else:
            print("[Planner] SceneRelevanceGraph checkpoint bulunamadi; mode_selector grafiksiz calisacak.")

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
                #print(occupancy)

        # Annotate max speed along the reference path
        target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
        #print(f"Ref_path dimensions: {ref_path.shape}")
        if len(ref_path) < MAX_LEN * 10:
            ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)
        
        return ref_path.astype(np.float32)
    
    def _get_reference_path_multi(self, ego_state, traffic_light_data, observation):
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

        if starting_block is None:
            if getattr(self, '_debug', False):
                print('[Planner] _get_reference_path_multi: starting_block=None')
            return None
            
        if closest_distance > 5:
            if getattr(self, '_debug', False):
                print(f'[Planner] _get_reference_path_multi: ego off-route (closest_distance={closest_distance:.2f}m)')
            return None

        # YENİ: Tek rota yerine rota listesi döndüren fonksiyonu çağırıyoruz
        try:
            # max_routes olarak 5 verdik
            ref_paths_list = self._path_planner.plan_multiple(ego_state, starting_block, observation, traffic_light_data, top_k=5)
        except Exception as e:
            if getattr(self, '_debug', False):
                print(f'[Planner] _get_reference_path_multi: plan_multiple exception: {type(e).__name__}: {e}')
            ref_paths_list = None

        if not ref_paths_list: # Liste boşsa veya None döndüyse
            if getattr(self, '_debug', False):
                print('[Planner] _get_reference_path_multi: empty ref_paths_list')
            return None

        multi_modal_paths = []
        
        # YENİ: Gelen her bir rota için ışık ve hız eklemelerini yap
        for ref_path in ref_paths_list:
            occupancy = np.zeros(shape=(ref_path.shape[0], 1))
            for data in traffic_light_data:
                id_ = str(data.lane_connector_id)
                if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                    lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                    if lane_conn:
                        conn_path = lane_conn.baseline_path.discrete_path
                        conn_path = np.array([[p.x, p.y] for p in conn_path])
                        red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                        occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

            target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
            target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
            max_speed = annotate_speed(ref_path, target_speed)

            # Birleştir: [x, y, theta, k, v_max, occupancy]
            annotated_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1)
            
            # Boyutu sabitle
            if len(annotated_path) < MAX_LEN * 10:
                annotated_path = np.append(annotated_path, np.repeat(annotated_path[np.newaxis, -1], MAX_LEN*10-len(annotated_path), axis=0), axis=0)
            
            multi_modal_paths.append(annotated_path.astype(np.float32))

        # Listeyi tek bir numpy matrisine dönüştür
        # Çıktı Shape: (N, MAX_LEN*10, 6) -> N en fazla 5 olacak
        return np.stack(multi_modal_paths, axis=0)
    
    def get_multimodal_reference_paths(self, ego_state, traffic_light_data, max_routes=5, points_per_route=MAX_LEN * 10, search_distance=150.0):
        import numpy as np
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
        
        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_heading = ego_state.rear_axle.heading
        ego_point = Point2D(ego_x, ego_y)
        
        # 1. GLOBAL ROTA FİLTRESİ (GameFormer Navigasyon Kısıtı)
        # Sadece hedefe giden yol bloklarındaki şeritlerin ID'lerini topluyoruz
        valid_route_edge_ids = set()
        for block in getattr(self, '_route_roadblocks', []):
            for edge in block.interior_edges:
                valid_route_edge_ids.add(edge.id)
                
        # Nihai çıktı matrisi: [N_lat, Seq_Len, Feature_Dim] -> Feature_Dim = 6
        # Features: [x, y, yaw, k (curvature), v_max, occupancy]
        c_lat_candidates = np.zeros((max_routes, points_per_route, 6), dtype=np.float32)
        
        # 2. BAŞLANGIÇ ŞERİTLERİNİ BUL
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        current_map_objs = self._map_api.get_proximal_map_objects(ego_point, 5.0, layers)
        
        valid_start_lanes = []
        if current_map_objs:
            for layer in layers:
                if layer in current_map_objs and current_map_objs[layer]:
                    for lane in current_map_objs[layer]:
                        # Sadece Global Rotaya dahil olan şeritlerden başla
                        if lane.id in valid_route_edge_ids:
                            lane_pts = lane.baseline_path.discrete_path
                            if len(lane_pts) > 0:
                                mid_pt = lane_pts[len(lane_pts)//2]
                                heading_diff = mid_pt.heading - ego_heading
                                if np.cos(heading_diff) > 0.5: # Yönelim farkı tolere edilebilir seviyede mi
                                    valid_start_lanes.append(lane)
                                    
        if not valid_start_lanes:
            return c_lat_candidates # Uygun şerit yoksa sıfır matrisi dön
            
        def dist_to_ego(lane):
            pts = lane.baseline_path.discrete_path
            return min((pt.x - ego_x)**2 + (pt.y - ego_y)**2 for pt in pts)
            
        valid_start_lanes.sort(key=dist_to_ego)
        queue = [(lane, [lane], 0.0) for lane in valid_start_lanes]
        candidate_paths = []
        
        # 3. KISITLANMIŞ BFS ARAMASI (Sadece hedefe doğru dallanma)
        while queue and len(candidate_paths) < max_routes:
            current_lane, path_lanes, path_length = queue.pop(0)
            
            lane_pts = current_lane.baseline_path.discrete_path
            lane_len = np.hypot(lane_pts[-1].x - lane_pts[0].x, lane_pts[-1].y - lane_pts[0].y) if len(lane_pts) > 1 else 0.0
            new_length = path_length + lane_len
            
            if new_length >= search_distance:
                candidate_paths.append(path_lanes)
                continue
                
            next_possible_lanes = []
            if current_lane.outgoing_edges:
                for next_lane in current_lane.outgoing_edges:
                    # KRİTİK FİLTRE: Sadece rotada olan ve döngü yapmayan şeritler
                    if next_lane.id in valid_route_edge_ids and next_lane.id not in [l.id for l in path_lanes]:
                        next_possible_lanes.append(next_lane)
                        
            if not next_possible_lanes:
                candidate_paths.append(path_lanes)
            else:
                for next_lane in next_possible_lanes:
                    queue.append((next_lane, path_lanes + [next_lane], new_length))
                    
        # 4. GEOMETRİK İNTERPOLASYON VE ZENGİNLEŞTİRME (ANNOTATION)
        c_rot, s_rot = np.cos(-ego_heading), np.sin(-ego_heading)
        
        for i, lane_path in enumerate(candidate_paths):
            if i >= max_routes:
                break
                
            full_centerline = []
            speed_limits = []
            lane_ids = set([l.id for l in lane_path]) # Trafik ışığı kontrolü için
            
            for lane in lane_path:
                speed_limit = lane.speed_limit_mps or getattr(self, '_target_speed', 15.0)
                speed_limit = np.clip(speed_limit, 3.0, 15.0)
                
                for point in lane.baseline_path.discrete_path:
                    dx = point.x - ego_x
                    dy = point.y - ego_y
                    
                    rel_x = dx * c_rot - dy * s_rot
                    rel_y = dx * s_rot + dy * c_rot
                    rel_yaw = point.heading - ego_heading
                    
                    if rel_x > -2.0: # Aracın arkasında kalanları kırp
                        full_centerline.append([rel_x, rel_y, rel_yaw])
                        speed_limits.append(speed_limit)
                        
            full_centerline = np.array(full_centerline)
            speed_limits = np.array(speed_limits)
            
            if len(full_centerline) > 1:
                # A. İnterpolasyon (X, Y, Yaw, V_max)
                orig_indices = np.linspace(0, 1, len(full_centerline))
                target_indices = np.linspace(0, 1, points_per_route)
                
                ref_x = np.interp(target_indices, orig_indices, full_centerline[:, 0])
                ref_y = np.interp(target_indices, orig_indices, full_centerline[:, 1])
                ref_yaw = np.interp(target_indices, orig_indices, full_centerline[:, 2])
                ref_vmax = np.interp(target_indices, orig_indices, speed_limits)
                
                # B. Eğrilik (Curvature - k) Hesaplama
                dx = np.gradient(ref_x)
                dy = np.gradient(ref_y)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
                curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
                
                # C. Trafik Işığı (Occupancy) Etiketleme
                occupancy = np.zeros(points_per_route)
                for data in traffic_light_data:
                    if data.status == TrafficLightStatusType.RED and str(data.lane_connector_id) in lane_ids:
                        lane_conn = self._map_api.get_map_object(str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
                        if lane_conn:
                            conn_path = lane_conn.baseline_path.discrete_path
                            if len(conn_path) > 0:
                                # Kırmızı ışığın başlangıç noktasını ego frame'e çevir
                                cx = conn_path[0].x - ego_x
                                cy = conn_path[0].y - ego_y
                                rel_cx = cx * c_rot - cy * s_rot
                                # Kırmızı ışıktan sonraki tüm noktaları geçilmez (occupancy=1) işaretle
                                occupancy[ref_x > rel_cx] = 1.0
                                
                # D. Modu Birleştir ve Kaydet
                c_lat_candidates[i] = np.stack([ref_x, ref_y, ref_yaw, curvature, ref_vmax, occupancy], axis=-1)
                
        return c_lat_candidates
    
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

    def get_multimodal_reference_paths2(self, ego_state, traffic_light_data, max_routes=5, points_per_route=1500, search_distance=150.0):
        """Build multimodal reference paths via LatticePlanner and return both ego/global frames."""
        import numpy as np
        import scipy.spatial
        from shapely import Point
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
        # Kendi yazdığınız fonksiyonu import ettiğinizi varsayıyorum
        # from Planner.planner_utils import annotate_occupancy 

        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_heading = ego_state.rear_axle.heading

        c_lat_candidates_ego = np.zeros((max_routes, points_per_route, 6), dtype=np.float32)
        c_lat_candidates_global = np.zeros((max_routes, points_per_route, 6), dtype=np.float32)

        # Collect route blocks and route edge ids exactly like planner-side route filtering.
        route_blocks = []
        route_edge_ids = []
        route_roadblock_ids = self._route_roadblock_ids if hasattr(self, '_route_roadblock_ids') else []
        map_api = self._map_api
        for id_ in route_roadblock_ids:
            block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if not block:
                continue
            route_blocks.append(block)
            for edge in block.interior_edges:
                route_edge_ids.append(edge.id)

        if not route_blocks or not route_edge_ids:
            return c_lat_candidates_ego, c_lat_candidates_global

        # Find the nearest block to the current ego pose.
        cur_point = (ego_x, ego_y)
        closest_distance = np.inf
        starting_block = None
        for block in route_blocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    closest_distance = distance
                    starting_block = block

            if np.isclose(closest_distance, 0):
                break

        if starting_block is None:
            return c_lat_candidates_ego, c_lat_candidates_global

        # Use LatticePlanner's native candidate edge/path generation.
        lattice_planner = LatticePlanner(route_edge_ids, max_len=search_distance)
        edges = lattice_planner.get_candidate_edges(starting_block, ego_state)
        candidate_paths = lattice_planner.get_candidate_paths(edges)
        if candidate_paths is None:
            return c_lat_candidates_ego, c_lat_candidates_global

        target_speed = starting_block.interior_edges[0].speed_limit_mps
        if target_speed is None:
            target_speed = 15.0

        c_rot, s_rot = np.cos(-ego_heading), np.sin(-ego_heading)
        
        # DÜZELTME 1: Geçerli rotaları sırayla dizmek için sayaç (Arada sıfır boşluk kalmasın)
        valid_count = 0

        for _, (_, _, lane_path, path_polyline) in candidate_paths:
            if valid_count >= max_routes:
                break

            lane_ids = set([str(l.id) for l in lane_path])

            full_centerline_ego = []
            full_centerline_global = []
            for p in path_polyline:
                dx = p[0] - ego_x
                dy = p[1] - ego_y
                rel_x = dx * c_rot - dy * s_rot
                rel_y = dx * s_rot + dy * c_rot
                rel_yaw = p[2] - ego_heading

                if rel_x > -2.0:
                    full_centerline_ego.append([rel_x, rel_y, rel_yaw])
                    full_centerline_global.append([p[0], p[1], p[2]])

            full_centerline_ego = np.array(full_centerline_ego)
            full_centerline_global = np.array(full_centerline_global)
            if len(full_centerline_ego) <= 1:
                continue

            # DÜZELTME 2: Fiziksel Mesafeye Göre İnterpolasyon (Viraj Koruması)
            dp = np.diff(full_centerline_ego, axis=0)
            segment_dists = np.hypot(dp[:, 0], dp[:, 1])
            cum_dists = np.insert(np.cumsum(segment_dists), 0, 0.0)
            
            target_dists = np.linspace(0, cum_dists[-1], points_per_route)
            
            ref_x_ego = np.interp(target_dists, cum_dists, full_centerline_ego[:, 0])
            ref_y_ego = np.interp(target_dists, cum_dists, full_centerline_ego[:, 1])
            ref_yaw_ego = np.interp(target_dists, cum_dists, full_centerline_ego[:, 2])

            ref_x_global = np.interp(target_dists, cum_dists, full_centerline_global[:, 0])
            ref_y_global = np.interp(target_dists, cum_dists, full_centerline_global[:, 1])
            ref_yaw_global = np.interp(target_dists, cum_dists, full_centerline_global[:, 2])

            # Curvature
            dx = np.gradient(ref_x_global)
            dy = np.gradient(ref_y_global)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

            max_speed = np.full(points_per_route, target_speed, dtype=np.float32)

            # DÜZELTME 3: Kendi Kusursuz 2D annotate_occupancy Fonksiyonunuzu Kullanım
            occupancy = np.zeros((points_per_route, 1))
            ego_path_2d = np.column_stack([ref_x_ego, ref_y_ego])
            
            for data in traffic_light_data:
                if data.status == TrafficLightStatusType.RED and str(data.lane_connector_id) in lane_ids:
                    lane_conn = map_api.get_map_object(str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
                    if not lane_conn:
                        continue

                    conn_path = lane_conn.baseline_path.discrete_path
                    if len(conn_path) == 0:
                        continue

                    # Kırmızı ışık hattını Ego Frame'e topluca çevir
                    conn_pts = np.array([[p.x, p.y] for p in conn_path])
                    dx_l = conn_pts[:, 0] - ego_x
                    dy_l = conn_pts[:, 1] - ego_y
                    rel_cx = dx_l * c_rot - dy_l * s_rot
                    rel_cy = dx_l * s_rot + dy_l * c_rot
                    red_light_lane_ego = np.column_stack([rel_cx, rel_cy])
                    
                    occupancy = annotate_occupancy(occupancy, ego_path_2d, red_light_lane_ego)

            occupancy_flat = occupancy.squeeze()

            # Matrislere sırayla (boşluk bırakmadan) ata
            c_lat_candidates_ego[valid_count] = np.stack([ref_x_ego, ref_y_ego, ref_yaw_ego, curvature, max_speed, occupancy_flat], axis=-1)
            c_lat_candidates_global[valid_count] = np.stack([ref_x_global, ref_y_global, ref_yaw_global, curvature, max_speed, occupancy_flat], axis=-1)
            
            valid_count += 1

        # NOT: Eğer 5'ten az rota dönerse, tensörün geri kalanı 
        # en başta np.zeros ile oluşturulduğu için sıfır kalmaya devam edecektir. (İsteğiniz üzere).
        
        return c_lat_candidates_ego, c_lat_candidates_global

    def _get_prediction(self, features):
        
        with torch.no_grad():
            # FAZ 1: Çevresel Özellik Çıkarımı (Backbone)
            encoder_outputs = self.backbone.encoder(features)
            route_lanes = encoder_outputs['route_lanes']
            initial_state = encoder_outputs['actors'][:, 0, -1]
            decoder_outputs, env_encoding = self.backbone.decoder(encoder_outputs)
            
            _ , ego_plan, neural_plan = self.backbone(features)


        # Diğer ajanların (komşuların) gelecekteki konumlarını çıkart
        K = len([k for k in decoder_outputs.keys() if 'interactions' in k]) - 1
        final_predictions = decoder_outputs[f'level_{K}_interactions'][:, 1:]
        final_scores = decoder_outputs[f'level_{K}_scores']
        
        ego_current = features['ego_agent_past'][:, -1]
        neighbors_current = features['neighbor_agents_past'][:, :, -1]
        
        return ego_plan, neural_plan, final_predictions, final_scores, ego_current, neighbors_current
    
    def finalize_selected_route(self, ego_state, chosen_global_6d, chosen_speed_mps, traffic_light_data):
        """
        Ağın seçtiği Global 6D rotayı alır, orijinal LatticePlanner fonsiyonlarıyla (Bezier+Spline) 
        işler ve Refiner için nihai Ego Frame 6D rotayı döner.
        """
        target_len = self._max_path_length * 10
        
        # 1. GERÇEK FİZİKSEL UZUNLUĞU BUL
        global_polyline = chosen_global_6d[:, :3]
        dp = np.diff(global_polyline, axis=0)
        ds = np.hypot(dp[:, 0], dp[:, 1])
        cum_dists = np.insert(np.cumsum(ds), 0, 0.0)
        total_len = cum_dists[-1]
        
        # --- KRİTİK KORUMA 1: ROTA SONU (İterasyon 129 Kilitlenmesini Önler) ---
        # Eğer yol 2 metreden kısaysa veya sadece 0'lardan oluşuyorsa
        if total_len < 2.0 or (np.allclose(global_polyline[:, 0], 0.0) and np.allclose(global_polyline[:, 1], 0.0)):
            rx = np.linspace(0.0, 10.0, target_len)
            ry = np.zeros(target_len)
            ryaw = np.zeros(target_len)
            rk = np.zeros(target_len)
            v_max = np.zeros(target_len) # HIZ = 0 (Güvenli Fren)
            occupancy = np.zeros(target_len)
            return np.column_stack([rx, ry, ryaw, rk, v_max, occupancy]).astype(np.float32)

        # 2. ORİJİNAL ÇÖZÜNÜRLÜĞE (FİZİKSEL SEYRELTME) GERİ DÖNÜŞ
        # NuPlan'ın orijinal fonksiyonlarını çökertmemek için noktaları fiziki olarak 0.5 metrede bir olacak şekilde ayarlıyoruz.
        num_points = max(3, int(total_len / 0.5)) 
        target_dists = np.linspace(0, total_len, num_points)
        
        rx = np.interp(target_dists, cum_dists, global_polyline[:, 0])
        ry = np.interp(target_dists, cum_dists, global_polyline[:, 1])
        ryaw = np.interp(target_dists, cum_dists, global_polyline[:, 2])
        
        mock_polyline = np.column_stack([rx, ry, ryaw])
        mock_paths = [(0, (0.0, 0.0, None, mock_polyline))]
        
        # 3. BEZIER İLE ARACA BAĞLAMA (Artık asla kilitlenmez)
        generated_paths = self._path_planner.generate_paths(ego_state, mock_paths)
        optimal_path_polyline = generated_paths[0][0]
        
        # --- KRİTİK KORUMA 2: SPLINE SINGULAR MATRIX ÖNLEYİCİ ---
        dp = np.diff(optimal_path_polyline, axis=0)
        ds = np.hypot(dp[:, 0], dp[:, 1])
        mask = np.insert(ds > 0.1, 0, True)
        optimal_path_polyline = optimal_path_polyline[mask]

        if len(optimal_path_polyline) < 3:
            rx = np.linspace(0.0, 10.0, target_len)
            ry = np.zeros(target_len)
            ryaw = np.zeros(target_len)
            rk = np.zeros(target_len)
            v_max = np.zeros(target_len)
            occupancy = np.zeros(target_len)
            return np.column_stack([rx, ry, ryaw, rk, v_max, occupancy]).astype(np.float32)

        # 4. SPLINE VE EGO FRAME DÖNÜŞÜMÜ
        # Mimic _get_reference_path: swallow any spline/post_process failure (e.g.
        # singular matrix on too-short polylines) and fall back to safe brake path.
        try:
            ref_path_4d = self._path_planner.post_process(optimal_path_polyline, ego_state)
        except Exception:
            rx = np.linspace(0.0, 10.0, target_len)
            ry = np.zeros(target_len)
            ryaw = np.zeros(target_len)
            rk = np.zeros(target_len)
            v_max = np.zeros(target_len)
            occupancy = np.zeros(target_len)
            return np.column_stack([rx, ry, ryaw, rk, v_max, occupancy]).astype(np.float32)
        
        # 5. HIZ VE IŞIK ENJEKSİYONU
        # Egrilik-tabanli (fizik) hiz profili: donus-sonrasi duz kesimde hiz geri toplanir
        # -> ego_progress_along_expert_route ve ego_is_making_progress metriklerini iyilestirir.
        # (Eski kor '3 m/s sabitle' davranisi annotate_speed'de duruyor; model-feature yollari
        #  satir 172/240 hala onu kullaniyor, sadece SURULEN trajektori burada degisti.)
        max_speed = annotate_speed_curvature(ref_path_4d, chosen_speed_mps)
        occupancy = np.zeros(shape=(ref_path_4d.shape[0], 1))
        
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                if lane_conn and len(lane_conn.baseline_path.discrete_path) > 0:
                    conn_path = np.array([[p.x, p.y] for p in lane_conn.baseline_path.discrete_path])
                    red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                    occupancy = annotate_occupancy(occupancy, ref_path_4d, red_light_lane)

        # 6. BİRLEŞTİRME
        final_ref_path = np.concatenate([ref_path_4d, max_speed, occupancy], axis=-1)
        
        if len(final_ref_path) < target_len:
            final_ref_path = np.append(
                final_ref_path, 
                np.repeat(final_ref_path[np.newaxis, -1], target_len - len(final_ref_path), axis=0), 
                axis=0
            )
            
        return final_ref_path.astype(np.float32)

    def _save_debug_plot(self, features, ref_path, best_c_lat, neural_plan, ego_plan, final_path, iteration=0):
        if not self._debug:
            return
        if self._debug_plot_count >= self._debug_max_plots:
            return

        # YENİ: Senaryoya özel klasörün içindeki "candidates" alt klasörüne kaydet
        out_dir = getattr(self, '_current_scenario_dir', self._debug_dir or "testing_log/debug_plots")
        out_dir = os.path.join(out_dir, "candidates")
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

        if best_c_lat is not None:
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

    def _record_relevance(self, features, iteration=0, lat_idx=None, lon_idx=None):
        """--debug: o adimin GAT onem verisini kompakt numpy record olarak biriktir ve
        senaryo klasorune relevance_data.pkl olarak yaz. render_relevance_video.py bundan
        senaryo videosu uretir. (Goruntu degil VERI kaydedilir.)"""
        graph_out = getattr(self, '_prev_importance', None)
        if (graph_out is None) or ('importance' not in graph_out):
            return
        rec = build_relevance_record(
            features, graph_out,
            extra={'iteration': int(iteration),
                   'lat_idx': None if lat_idx is None else int(lat_idx),
                   'lon_idx': None if lon_idx is None else int(lon_idx)},
        )
        self._relevance_records.append(rec)
        out_dir = getattr(self, '_current_scenario_dir', self._debug_dir or "testing_log/debug_plots")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'relevance_data.pkl'), 'wb') as f:
            pickle.dump({'threshold': self._relevance_threshold, 'records': self._relevance_records}, f)

    def _save_candidates_debug_plot(self, features, candidates, best_idx=None, iteration=0,
                                     top3=None, num_lon=12):
        """
        top3: optional list of 3 tuples [(mode_idx, score), ...] sorted by rank (0 = selected).
              mode_idx in [0, num_lat*num_lon). lat = mode_idx // num_lon, lon = mode_idx % num_lon.
              Each mode maps to a (c_lat route, target speed) pair.
        """
        if not self._debug:
            return
        if candidates is None:
            return
        if self._debug_candidates_plot_count >= self._debug_max_plots:
            return

        out_dir = getattr(self, '_current_scenario_dir', self._debug_dir or "testing_log/debug_plots")
        out_dir = os.path.join(out_dir, "candidates")
        os.makedirs(out_dir, exist_ok=True)

        # Tek panel BEV; GAT onemi (esigi gecenler) dogrudan bu haritanin uzerine bindirilir
        graph_out = getattr(self, '_prev_importance', None)
        has_graph = (graph_out is not None) and ('importance' in graph_out)
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)

        # Harita ve geçmiş ajanları çiz
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

        # --- Yeni tasarim: yildiz tabanli top-3 ---
        # Her top-k mod = (route, hiz). 8 sn sonra ego'nun olacagi konum:
        #   pos_8s = route_polyline @ distance = speed * 8s
        # Bu konuma renkli yildiz koyariz. Ayni rota + farkli hiz -> ayni cizgi
        # uzerinde 3 farkli pozisyonda yildiz -> SPEED FARKLARI GORSEL OLUR.
        rank_colors  = {0: 'tab:orange', 1: 'mediumpurple', 2: 'darkcyan'}
        rank_sizes   = {0: 320, 1: 200, 2: 130}
        rank_zs      = {0: 14,  1: 13,  2: 12}
        rank_titles  = {0: 'TOP-1 (SELECTED)', 1: 'TOP-2', 2: 'TOP-3'}
        HORIZON_S = 8.0
        ROUTE_PT_SPACING_M = 0.1  # candidates 0.1m spacing ile sample edilmis

        # --- Aday rotalari ciz: her top-k icin AYRI olarak kendi renginde ciz ---
        # Ayni lat'a birden fazla rank duserse her biri ayri kez cizilir (z-order'la):
        # en yuksek rank en ustte, daha kalin. Stars zaten farkli pozisyonda olur.
        rank_route_lw    = {0: 4.0, 1: 3.0, 2: 2.2}
        rank_route_z     = {0: 8,   1: 7,   2: 6}

        # Her rota icin son geçerli nokta indeksini hesapla (padded sifir bolgesini sayma)
        def _last_valid_idx_of(path):
            """6 ozelliklik path icin tam-sifir noktalari padding sayar."""
            valid = (np.abs(path).sum(axis=-1) > 1e-4)
            valid_idx = np.where(valid)[0]
            return int(valid_idx[-1]) if valid_idx.size > 0 else -1

        # 1) Once gri arka plan adaylari (top-3'e dahil olmayanlar)
        top3_lats = set()
        if top3 is not None:
            top3_lats = {mode_idx // num_lon for (mode_idx, _) in top3}

        drawn_count = 0
        for i in range(candidates.shape[0]):
            path = candidates[i]
            if path.ndim != 2 or path.shape[1] < 2:
                continue
            last_v = _last_valid_idx_of(path)
            if last_v < 1:
                continue
            xy = path[:last_v + 1, :2]
            if i not in top3_lats:
                ax.plot(xy[:, 0], xy[:, 1], color='gray', linewidth=1.4,
                        alpha=0.5, linestyle='--', zorder=5)
            drawn_count += 1

        # 2) Sonra her top-k rotasini AYRI olarak ciz (top-3'ten top-1'e dogru, top-1 en ustte)
        if top3 is not None:
            for rank in [2, 1, 0]:
                mode_idx, _ = top3[rank]
                lat = mode_idx // num_lon
                if lat < 0 or lat >= candidates.shape[0]:
                    continue
                path = candidates[lat]
                last_v = _last_valid_idx_of(path)
                if last_v < 1:
                    continue
                xy = path[:last_v + 1, :2]
                ax.plot(xy[:, 0], xy[:, 1], color=rank_colors[rank],
                        linewidth=rank_route_lw[rank], alpha=0.95,
                        zorder=rank_route_z[rank])

        if drawn_count == 0:
            ax.text(0.5, 0.5, 'No valid candidates', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)

        # --- Her top-k icin yildiz ekle (target_idx'i son geçerli noktaya clamp et) ---
        if top3 is not None:
            for rank, (mode_idx, score) in enumerate(top3):
                lat = mode_idx // num_lon
                if lat < 0 or lat >= candidates.shape[0]:
                    continue
                lon = mode_idx % num_lon
                speed_mps = lon / 11.0 * 15.0
                target_dist_m = speed_mps * HORIZON_S
                target_idx = int(target_dist_m / ROUTE_PT_SPACING_M)

                path = candidates[lat]
                last_v = _last_valid_idx_of(path)
                if last_v < 1:
                    continue
                # target_idx pad bolgesine duserse son geçerli noktaya kaydir
                target_idx = min(target_idx, last_v)
                px, py = float(path[target_idx, 0]), float(path[target_idx, 1])

                ax.scatter(px, py, color=rank_colors[rank], s=rank_sizes[rank],
                           marker='*', edgecolors='black', linewidths=1.5,
                           zorder=rank_zs[rank])

        ax.scatter([0.0], [0.0], marker='x', s=60, color='black', zorder=15)

        # --- Title sade ---
        if top3 is not None:
            mode_idx_0, score_0 = top3[0]
            speed_0 = (mode_idx_0 % num_lon) / 11.0 * 15.0
            ax.set_title(
                f'Mode Selector top-3 — Iter {iteration}   '
                f'(SELECTED v={speed_0:.1f} m/s, score={score_0:.2f})',
                fontsize=10,
            )
        else:
            ax.set_title(f'Candidate Reference Paths — Iter {iteration}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # --- Legend ---
        legend_handles = [
            Line2D([0], [0], color='c', lw=3, label='lanes'),
            Line2D([0], [0], color='b', lw=4, label='crosswalks'),
            Line2D([0], [0], color='g', lw=4, label='route_lanes'),
            Line2D([0], [0], color='#00a8e8', lw=2, label='ego_past'),
            Line2D([0], [0], color='m', lw=2, label='neighbors_past'),
            Line2D([0], [0], color='gray', lw=1.4, linestyle='--', label='other candidate routes'),
        ]
        if top3 is not None:
            for rank in range(3):
                mode_idx, score = top3[rank]
                lat_r = mode_idx // num_lon
                v = (mode_idx % num_lon) / 11.0 * 15.0
                legend_handles.append(
                    Line2D([0], [0], marker='*', color=rank_colors[rank],
                           markerfacecolor=rank_colors[rank], markeredgecolor='black',
                           markersize=14 - rank * 2, lw=rank_route_lw[rank],
                           label=f'{rank_titles[rank]}: lat={lat_r} @ v={v:.1f} m/s (score={score:.2f})')
                )

        ax.legend(handles=legend_handles, loc='best', fontsize=8)

        # --- ONEM OVERLAY: normalize onemi esigi gecen eleman/ajanlar bu BEV uzerinde vurgulanir ---
        if has_graph:
            rec = build_relevance_record(features, graph_out)
            n_pass = draw_relevance(ax, rec, threshold=self._relevance_threshold, draw_base=False)
            ax.set_title(ax.get_title() + f'  | onem>={self._relevance_threshold:.2f}: {n_pass} eleman (kirmizi)')

        file_name = os.path.join(out_dir, f'debug_candidates_iter_{iteration:04d}.png')
        fig.savefig(file_name, dpi=120, bbox_inches='tight')
        plt.close(fig)
        self._debug_candidates_plot_count += 1
   
    def _plan(self, ego_state, history, traffic_light_data, observation, iteration=None):
        # Construct input features
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # 2. YENİ: Aday rotaları bul ve tensör olarak features içine ekle
        c_lat_candidates, c_lat_candidates_global = self.get_multimodal_reference_paths2(
            ego_state,
            traffic_light_data,
            points_per_route=MAX_LEN * 10,
        )
        # Adaylari uzamsal sol->sag sirala (ordinal + frame'ler arasi kararli indeks).
        # Egitimdeki DrivingData ile AYNI helper -> train/inference tutarli; lat_idx artik
        # kararli bir fiziksel serite karsilik gelir (jitter + indeks-cache riski azalir).
        c_lat_candidates, c_lat_candidates_global = sort_candidates_by_lateral(
            c_lat_candidates, c_lat_candidates_global
        )
        #print(c_lat_candidates.shape)  # (N, Seq_Len, 6)
        #features['c_lat_candidates'] = torch.tensor(c_lat_candidates, dtype=torch.float32, device=self._device).unsqueeze(0)
        #expert_c_lat_np = c_lat_candidates[0, 10:, :3]
        
        #ref_path_candidate = debug_candidate_paths[0, : , :] 
        # Get reference path
        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)
        #ref_path=debug_candidate_paths[0, :, :]
        #candidate_paths[0]

        #ref_path_smoot = self.apply_spline_for_refiner(ref_path, 10.0)
    

        # ModeSelector skoru en yuksek olani sec
        # Stabilite icin sadece her N iterasyonda bir yeniden secim yap, arada cache kullan
        # DEBUG MODE: her frame yeniden secimi calistir ki viz guncel kalsin
        SELECT_EVERY = 1
        num_lon = 12
        best_c_lat_np = None
        run_selector = (
            (iteration is None)
            or (iteration % SELECT_EVERY == 0)
            or (not hasattr(self, '_prev_lat_idx'))
            or self._debug  # debug'da her frame fresh top-3
        )

        if c_lat_candidates is not None:
            if run_selector:
                with torch.no_grad():
                    encoder_outputs = self.backbone.encoder(features)

                    # Decoder'i de calistir, komsular icin argmax-mod top1 yorungelerini cek
                    N_NBR = 10
                    decoder_outputs, _ = self.backbone.decoder(encoder_outputs)
                    last_k = max(int(k.split('_')[1]) for k in decoder_outputs if 'interactions' in k)
                    inter = decoder_outputs[f'level_{last_k}_interactions'][:, 1:1 + N_NBR]   # [B, N, M, T, 4]
                    sc = decoder_outputs[f'level_{last_k}_scores'][:, 1:1 + N_NBR]            # [B, N, M]
                    bm = sc.argmax(-1)
                    B_, N_, M_, T_, _ = inter.shape
                    g = bm.view(B_, N_, 1, 1, 1).expand(-1, -1, 1, T_, 2)
                    top1_fut = torch.gather(inter[..., :2], 2, g).squeeze(2)                  # [B, N, T, 2]
                    nbr_states = encoder_outputs['actors'][:, 1:1 + N_NBR, -1]                # [B, N, 5]
                    nbr_valid = ~encoder_outputs['mask'][:, 1:1 + N_NBR]                      # [B, N]

                    # SceneRelevanceGraph (varsa): rafine dugumler + onem dagilimini hesapla
                    graph_kwargs = {}
                    if self.relevance_graph is not None:
                        graph_out = self.relevance_graph(encoder_outputs, features, num_agents=N_NBR + 1,
                                                         return_attention=True)
                        graph_kwargs = dict(
                            graph_context=graph_out['context'],
                            graph_valid=graph_out['valid'],
                            importance=graph_out['importance'],
                            hard_topk=self._graph_hard_topk,
                        )
                        self._prev_importance = graph_out  # viz icin sakla

                    # c_lat_candidates: (N_lat, N_r, 6) -> [B, N_lat, N_r, feat]
                    c_lat_tensor = torch.tensor(c_lat_candidates, dtype=torch.float32, device=self._device).unsqueeze(0)
                    mode_scores, _ = self.planner_head(
                        encoder_outputs['encoding'],
                        c_lat_tensor,
                        scene_mask=encoder_outputs['mask'],
                        neighbor_top1_futures=top1_fut,
                        neighbor_current_states=nbr_states,
                        neighbor_valid=nbr_valid,
                        **graph_kwargs,
                    )
                    # Top-3 modlari yakala (viz icin) — rank 0 = secilen
                    top3_scores, top3_idx = mode_scores.topk(3, dim=1)
                    top3_idx_list = top3_idx[0].cpu().tolist()       # [3]
                    top3_scores_list = top3_scores[0].cpu().tolist() # [3]
                    self._prev_top3 = list(zip(top3_idx_list, top3_scores_list))

                    best_idx = top3_idx_list[0]
                    lat_idx = best_idx // num_lon
                    lon_idx = best_idx % num_lon
                    self._prev_lat_idx = lat_idx
                    self._prev_lon_idx = lon_idx
            else:
                # Cache'den onceki secimi kullan
                lat_idx = self._prev_lat_idx
                lon_idx = self._prev_lon_idx

            best_c_lat_np = c_lat_candidates[lat_idx, :, :3]
        
        chosen_global_route = c_lat_candidates_global[lat_idx]
        best_speed_mps = (lon_idx / 11.0) * 15.0

        # O MUHTEŞEM FONKSİYONU ÇAĞIR
        ref_path2 = self.finalize_selected_route(
            ego_state, 
            chosen_global_route, 
            best_speed_mps, 
            traffic_light_data
        )

        # Infer prediction model
        with torch.no_grad():
            ego_plan, neural_plan, predictions, scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features)

        # --- TEST: NeuralPlanner'i bypass et, ref_path2 + mode_selector'in lon_idx hizini kullan ---
        # neural_plan'in TrajectoryPlanner icindeki rolu: ref_path uzerinde baslangic speed
        # profili belirlemek. Burada o profili dogrudan constant-speed sampling ile uretiyoruz.
        # ref_path2 spacing = 0.1m, fake_plan_idx[t] = t * best_speed_mps (cunku v*t*0.1m / 0.1m).
        USE_FAKE_NEURAL_PLAN = False
        if USE_FAKE_NEURAL_PLAN and ref_path2 is not None:
            N_FAKE = 80   # 8s @ 10Hz
            DT_FAKE = 0.1
            fake_distances_m = np.arange(N_FAKE) * DT_FAKE * best_speed_mps
            fake_idx = (fake_distances_m * 10).astype(np.int32).clip(0, len(ref_path2) - 1)
            fake_plan_np = ref_path2[fake_idx, :3]   # [80, 3]: x, y, theta
            neural_plan = torch.from_numpy(fake_plan_np).float().unsqueeze(0).to(self._device)
        # --------------------------------------------------------------------------------

        # Trajectory refinement
        with torch.no_grad():
            final_plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
                                                 predictions, neural_plan, scores, ref_path2, observation)


        
        # 6. Debug Çizimi
        if self._debug:
            self._save_candidates_debug_plot(
                features,
                c_lat_candidates,
                best_idx=lat_idx, # <--- SEÇİLEN ROTAYI FONKSİYONA GÖNDERİYORUZ
                iteration=0 if iteration is None else iteration,
                top3=getattr(self, '_prev_top3', None),  # [(mode_idx, score), x3]
                num_lon=num_lon,
            )
            self._save_debug_plot(
                features=features, # HARİTA VERİSİ EKLENDİ
                ref_path=ref_path2,
                best_c_lat=best_c_lat_np,  # ModeSelector'dan seçilen rota (veya fallback)
                neural_plan=None, #neural_plan[0].detach().cpu().numpy(),
                ego_plan= None, #ego_plan[0].detach().cpu().numpy() if ego_plan is not None else None,
                final_path=final_plan,
                iteration=0 if iteration is None else iteration,
            )
            # Per-adim onem verisini diske biriktir -> sonra render_relevance_video.py ile video
            self._record_relevance(features, iteration=0 if iteration is None else iteration,
                                   lat_idx=lat_idx, lon_idx=lon_idx)

        # Çıktıları numpy'a çevir (Eski kodlar)
        
        states = transform_predictions_to_states(final_plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)

        return trajectory
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        trajectory = self._plan(ego_state, history, traffic_light_data, observation, iteration=iteration)
        print(f'Iteration {iteration}: {time.time() - s:.3f} s')  # Commented out: verbose iteration logging

        return trajectory
