import torch
from .predictor_modules import *


class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self._lane_len = 50
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        self.agent_encoder = AgentEncoder(agent_dim=11)
        self.ego_encoder = AgentEncoder(agent_dim=7)
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)

    def forward(self, inputs):
        # agents
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # vector maps
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks']

        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        input = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)

        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'agent_tokens': encoded_actors,  # FUSION ONCESI temiz per-ajan gommeleri (SceneRelevanceGraph node'lari icin)
            'mask': mask,
            'route_lanes': inputs['route_lanes'],
            'map_lanes': inputs['map_lanes'] # add this to get map data in physic based encocer
        }

        return encoder_outputs

class Decoder(nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(Decoder, self).__init__()
        self.levels = levels
        future_encoder = FutureEncoder()

        # initial level
        self.initial_predictor = InitialPredictionDecoder(modalities, neighbors)
        

        # level-k reasoning
        self.interaction_stage = nn.ModuleList([InteractionDecoder(modalities, future_encoder) for _ in range(levels)])

    def forward(self, encoder_outputs):
        decoder_outputs = {}
        current_states = encoder_outputs['actors'][:, :, -1]
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']

        # level 0 decode
        last_content, last_level, last_score = self.initial_predictor(current_states, encoding, mask)
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_score
        
        # level k reasoning
        for k in range(1, self.levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_score
        
        env_encoding = last_content[:, 0]

        return decoder_outputs, env_encoding

class DecoderP(nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(DecoderP, self).__init__()
        self.levels = levels
        self.neighbors = neighbors # <--- 1. DEĞİŞİKLİK: Bunu ekle (limit koymak için lazım)
        
        future_encoder = FutureEncoder()

        # initial level
        # self.initial_predictor = InitialPredictionDecoder(modalities, neighbors) # Eskisini kapattık
        self.initial_physics_predictor = PhysicsBasedDecoderv2(modalities=modalities)

        # level-k reasoning
        self.interaction_stage = nn.ModuleList([InteractionDecoder(modalities, future_encoder) for _ in range(levels)])

    def forward(self, encoder_outputs):
        decoder_outputs = {}
        
        # Tüm ajanların durumunu al
        full_current_states = encoder_outputs['actors'][:, :, -1] 
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']

        # <--- 2. DEĞİŞİKLİK: Ajan sayısını sınırla (Ego + Neighbors)
        # Eğer bunu yapmazsan 20 komşu hesaplar ama sistem 10 beklediği için hata verir.
        target_agent_count = self.neighbors + 1
        current_states = full_current_states[:, :target_agent_count]
        # -----------------------------------------------------------

        # Take map info
        map_lanes = encoder_outputs['map_lanes']

        # level 0 decode (Physics/Lane Following)
        # Artık sınırlandırılmış 'current_states' gönderiyoruz
        last_content, last_level, last_score = self.initial_physics_predictor(current_states, encoding, mask, map_lanes)

        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_score
        
        # level k reasoning
        for k in range(1, self.levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_score
        
        env_encoding = last_content[:, 0]

        return decoder_outputs, env_encoding

class NeuralPlanner(nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_fusion = CrossTransformer()
        self.plan_decoder = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*2))
        self.route_encoder = VectorMapEncoder(3, 50)

    def dynamics_layer(self, controls, initial_state):       
        dt = 0.1 # discrete time period [s]
        max_a = 5 # vehicle's accleration limits [m/s^2]
        max_d = 0.5 # vehicle's steering limits [rad]
        
        vel_init = torch.hypot(initial_state[..., 3], initial_state[..., 4])
        vel = vel_init[:, None] + torch.cumsum(controls[..., 0].clamp(-max_a, max_a) * dt, dim=-1)
        vel = torch.clamp(vel, min=0)

        yaw_rate = controls[..., 1].clamp(-max_d, max_d) * vel
        yaw = initial_state[:, None, 2] + torch.cumsum(yaw_rate * dt, dim=-1)
        yaw = torch.fmod(yaw, 2*torch.pi)

        vel_x = vel * torch.cos(yaw)
        vel_y = vel * torch.sin(yaw)

        x = initial_state[:, None, 0] + torch.cumsum(vel_x * dt, dim=-1)
        y = initial_state[:, None, 1] + torch.cumsum(vel_y * dt, dim=-1)

        return torch.stack((x, y, yaw), dim=-1)

    def forward(self, env_encoding, route_lanes, initial_state):
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        env_route_encoding = torch.max(env_route_encoding, dim=1)[0] # max pooling over modalities
        control = self.plan_decoder(env_route_encoding)
        control = control.reshape(-1, self._future_len, 2)
        plan = self.dynamics_layer(control, initial_state)

        return plan


class GameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        #self.mode_selector = ModeSelector()
        self.planner = NeuralPlanner()

    def forward(self, inputs):
        #c_lat_candidates = inputs['c_lat_candidates']
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs['route_lanes']
        initial_state = encoder_outputs['actors'][:, 0, -1]
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        neural_plan = self.planner(env_encoding, route_lanes, initial_state)
        

        '''
        # --- MODE SELECTION ---
        mode_scores, mode_features = self.mode_selector(env_encoding, c_lat_candidates)
        
        # --- 2. SADECE 1 TANE MOD SEÇ ---
        if forced_mode_idx is not None:
            # Eğitimde: Bizim verdiğimiz uzman modu
            selected_mode_idx = forced_mode_idx
        else:
            # Testte/Validasyonda: Ağın en yüksek puan verdiği 1 mod
            selected_mode_idx = torch.argmax(mode_scores, dim=1)
            
        B = mode_features.shape[0]
        batch_indices = torch.arange(B)
        
        # Seçilen tek bir modun özelliklerini çıkar -> [B, 256]
        # DİKKAT: .unsqueeze(1) işlemi SİLİNDİ. Dümdüz 2D tensör.
        best_mode_features = mode_features[batch_indices, selected_mode_idx]
        
        # --- 3. NEURAL PLANNER'A GÖNDER ---
        # neural_plan direkt olarak [B, 80, 3] döner. 
        neural_plan = self.planner(env_encoding, route_lanes, initial_state, best_mode_features)
        
        # Görselleştirme için rotayı çek
        lat_indices = selected_mode_idx // 12  
        best_c_lat = c_lat_candidates[batch_indices, lat_indices]
        '''
        # --- EGO PLAN (DECODER) ÇIKARIMI ---
        level_k = max([int(k.split('_')[1]) for k in decoder_outputs.keys() if 'interactions' in k])
        ego_trajectories = decoder_outputs[f'level_{level_k}_interactions'][:, 0]  
        ego_scores = decoder_outputs[f'level_{level_k}_scores'][:, 0]  
        
        best_mode_idx = torch.argmax(ego_scores, dim=-1)  
        ego_xy = ego_trajectories[torch.arange(ego_trajectories.size(0)), best_mode_idx, :, :2]  
        
        ego_dxy = torch.zeros_like(ego_xy)
        ego_dxy[:, 1:] = ego_xy[:, 1:] - ego_xy[:, :-1]
        # 2. SMOOTH
        smooth_dxy = ego_dxy.clone()
        smooth_dxy[:, 1:-1] = (ego_dxy[:, :-2] + ego_dxy[:, 1:-1] + ego_dxy[:, 2:]) / 3.0
        smooth_dxy[:, 0] = smooth_dxy[:, 1]
        smooth_dxy[:, -1] = smooth_dxy[:, -2]
        raw_heading = torch.atan2(smooth_dxy[:, :, 1], smooth_dxy[:, :, 0])  
        step_distances = torch.norm(smooth_dxy, dim=-1)
        stopped_mask = step_distances < 0.05
        initial_yaw = initial_state[:, 2].unsqueeze(1).expand(-1, ego_xy.size(1))
        ego_heading = torch.where(stopped_mask, initial_yaw, raw_heading)
        ego_plan = torch.stack([ego_xy[:, :, 0], ego_xy[:, :, 1], ego_heading], dim=-1)

        
        return decoder_outputs, ego_plan, neural_plan
    