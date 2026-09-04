"""L0 v3 -- CHANNELS_AUDIT v3 sozlugu.

channels.py'nin YERINE GECMEZ; yaninda durur. Eski checkpoint'ler ve eski sonuclar
channels.py ile uretilmeye devam eder. Bu modul yeni npz alanlarini kullanir:
    lane_tl        [B,L,P,4]  DOGRU trafik isigi one-hot (data_process list() duzeltmesi)
    intersections  [B,I,P,3]  kavsak poligonlari, ego-frame, (x,y,dolu)
    stop_polygons  [B,S,P,3]  dur cizgileri, ego-frame

Sozlukteki degisiklikler (gerekce icin CHANNELS_AUDIT.md v3):
  AJAN  near CIKTI (1061 ajanin 1058'i GT'de none) · follows/merges/overtakes L1'e ·
        onObservedCollisionCourseWith L1'e · inCrosswalk, staticObstacleOnPath,
        onRouteCorridor, sharedTrafficControl YENI · sharesIntersectionWith artik
        GERCEK kavsak poligonuyla (eskiden gelecek-tabanli proxy idi)
  HARITA near CIKTI (KG'de harita karsiligi yok) · inIntersection artik DOLU ·
        inCrosswalk YENI · stopLine trafficControl'e birlesti
"""
import math
import torch

from .channels import (_wrap, _project, _corridor_arrays, _arc_walk,
                       LANE_W, SAME_FLOW_RAD, MAP_DIR_RAD, DT)

# ---------------- L0 AJAN (6) ----------------
A_SAME_LANE_AHEAD = 0
A_SAME_LANE_BEHIND = 1         # YENIDEN ACILDI (2026-09-03): eski -25 m tanimi
                               # nedensellik-ters diye kapatilmisti; kullanici karariyla
                               # DAR menzille (ds >= -LON_BEHIND_M = 10 m) geri geldi --
                               # sadece tamponundaki takipci, uzak kuyruk degil.
A_LEFT_ADJACENT = 2
A_RIGHT_ADJACENT = 3
A_SHARES_INTERSECTION = 4      # np:sharesIntersectionWith -- GERCEK poligon
A_VRU_NEAR_PATH = 5            # reason vulnerable_road_user_near_ego_path
NUM_A = 6
A_NAMES = ["same_lane_ahead", "same_lane_behind", "leftAdjacent", "rightAdjacent",
           "sharesIntersectionWith", "VRU_near_ego_path"]

# --------------------------------------------------------------------------
# KAPATILDI: sharedTrafficControl  --  KG reason "shared_traffic_control"
#
# KG TASARIMI (relevance.py:12,124-131 + relevance_logic.py:252-254):
#     light_connectors   = {isik tasiyan lane_connector id'leri, O KARE}
#     entity_connectors  = {primary_map_object_id} U outgoing_object_ids
#     relevant_light     = bool(light & subject_connectors & object_connectors)
#     kapi: center <= traffic_control_distance_m = 45 m, skor +55, oncelik 60
#   Yani UCLU kimlik kesisimi: AYNI connector hem sinyalli, hem ego'nun
#   mevcut-veya-sonraki, hem ajanin.
#
# YAPAMIYORUZ: npz'de ne serit ID'si var, ne connector baglantisi, ne de
#   hangi connector'un sinyalli oldugu. Bizim uygulamamiz "ajan bir TL seridine
#   <= 3 m" idi -- "shared" kismi hic yoktu. Olculdu:
#     bizim yanma orani            : %16.0
#     KG'nin yanma orani           : % 0.4   (27393 cift-karede 119)
#     bizim yanmalarin ego ile kontrol PAYLASANI: %12.9
#   Yani 40 kat sisik. Duzeltilse bile ~%2.1, hala KG'nin 5 kati.
#
# NEDEN YENIDEN EXTRACTION'A DEGMEZ:
#   (a) KG'de %0.4 -- 27 bin cift-karede 119 tane, sinyal yok denecek kadar az.
#   (b) Ucli kesisim geregi sadece ego ile AYNI connector'daki, yani ego'nun
#       kendi seridindeki onundeki arac yaniyor (ayni kirmizida yan yana duran
#       iki arac paralel connector'larda, ID'leri farkli, yanmiyor). Bu da
#       buyuk olcude same_lane_ahead'in zaten yakaladigi kume.
#   (c) KG'nin KENDI oncelik tablosu bunu same_lane_ahead'in ALTINA koyuyor
#       (relevance_logic.py:293-308: shared_traffic_control 60, same_lane_ahead 75).
#
# GERI GETIRME KOSULU: npz'ye lane_ids + lane_outgoing_ids + agent_lane_id
#   eklemek, yani bir data_process degisikligi + tam yeniden extraction.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# KAPATILDI: inCrosswalk (ajan tarafi)  --  KG np:inCrosswalk
#
# ONCE: uygulama adiyla ortusmuyordu. Gecerli tanimi "ajan HERHANGI bir gecidin
#   SINIR NOKTALARINA <= 2.0 m" VE "sahnede ego yolunda BIR gecit var" idi:
#     near_cw    = d_cw.min() <= CROSSWALK_TOL          # poligon-ici testi DEGIL
#     cw_on_path = (...).any(-1)                        # [B] -- SAHNE bazli, gecit bazli degil
#   Olculdu (600 sahne, 192 yanan ajan):
#     gercekten bir gecit poligonunun ICINDE :  85 (%44.3)
#     hicbir poligonun icinde DEGIL          : 107 (%55.7)
#     poligon icinde AMA yanmayan            :  92
#     yanan ajanin EN YAKIN gecidi ego yolunda DEGIL : 76 (%39.6)
#   Yani kesinlik %44, duyarlilik %48. Gecitlerin medyan kosegeni 14.6 m oldugu icin
#   tam ortasinda duran yaya her sinir noktasindan >2 m uzakta kaliyor ve KACIYOR;
#   1.5 m disaridaki ise YANIYOR.
#
# SONRA: dogru tanimla (poligon-ici & O gecit ego koridorunda ileride) olculdu
#   (800 sahne):
#     yanan ajan 86 (cift basina ~%0.9);  31'i (%36.0) zaten VRU_near_ego_path'te
#     SADECE inCrosswalk olan 55: arac 26, yaya 29
#     gecidin ego koridorundaki mesafesi medyan 11.8 m -- %49'u VRU menzili (12 m) disinda
#   Yani duzeltilmis haliyle bile bagimsiz katkisi ~14 ajan/800 sahne. Geciteki ARAC
#   ego seridindeyse zaten same_lane_ahead; gecitteki YAYA'nin yarisi zaten VRU.
#   "Gecidin icinde olmak" ayri bir KARAR bilgisi tasimiyor -- sadece o VRU'nun
#   nerede durdugunu soyluyor.
#
# GERI GETIRME YERINE ONERILEN: kaybedilen tek gercek vaka (gecitte 20-30 m ileride
#   yaya) ayri bir kanal degil, VRU_near_ego_path'in MENZIL sorunu. Kanal eklemek
#   yerine VRU'yu koridor boyunca uzatmak daha temiz:
#     su an     : d_center <= VRU_MAX_M (12 m)
#     alternatif: d_center <= 12  VEYA  (koridorda & ds <= 30)
#   Bu ayri bir karar, henuz uygulanmadi.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# KAPATILDI: staticObstacleOnPath  (KG reason "static_obstacle_on_ego_path")
#
# NEDEN: bagimsiz katkisi yok. Olculdu (600 sahne):
#     P(same_lane_ahead | staticObstacleOnPath) = 95.4%
#   Tanim geregi zaten oyle: inlane & ds>0 & ds<=60 & speed<=0.5, yani
#   same_lane_ahead'in "duran" alt kumesi.
#
#   Yakan 174 ajanin tipi: arac 165 (%94.8), yaya 8 (%4.6), bisiklet 1 (%0.6).
#   same_lane_ahead'in YAKALAMADIGI 8 ajanin hepsi YAYA -- ve hepsi 12 m'den uzak
#   (yakin olanlari VRU_near_ego_path zaten aliyor: ego seridinde |d_lat|<=1.75 m,
#   ds<=12 m olan 15 VRU'nun 12'si yaniyor; kacan 3'u yay-mesafesi / kus-ucusu
#   metrik farkindan, sistematik boslук degil). Kullanici karari: 12 m'nin
#   otesindeki duran yaya ego planini etkilemiyor, kanal ona gerek duymuyor.
#
# ASIL SEBEP -- VERIDE DEBRIS YOK: neighbor_agents_past[..., 8:11] sadece 3 slot
#   (arac / yaya / bisiklet). Olculdu: 3825 gecerli ajanin HEPSI bu ucten birine
#   dusuyor, sinifsiz ajan 0. Yani koni, bariyer, czone_sign, generic_object
#   egitim tensorune HIC girmiyor -- bu kanal fiilen "duran arac" demek.
#   nuPlan'da bu nesneler VAR ve kendi kodumuz deployment'ta zaten sorguluyor
#   (Planner/state_lattice_path_planner.py:88-90).
#
# GERI GETIRME KOSULU: o dort nesne tipini egitim tensorune eklemek, yani BIR
#   data_process degisikligi daha + tam yeniden extraction. O yapilirsa kanal
#   gercekten bagimsiz bir kavram olur ("yolda duran, arac olmayan engel").
#   Yapilmazsa en fazla same_lane_ahead'in refinement'i olarak tutulabilir
#   (duran lider serit degisimi gerektirir, hareketli lider takip).
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# TARIHCE: same_lane_behind once KAPATILDI, sonra DAR menzille YENIDEN ACILDI.
# Asagidaki kapatma gerekcesi -25 m'lik ESKI tanim icindi; guncel tanim
# ds >= -LON_BEHIND_M (10 m) -- sadece tampondaki takipci.
# (KG: directional_same_lane_behind_m = 25 m; biz bilincli olarak daraltiyoruz.)
#
# NEDEN: nedensellik YONU ters. Ego'nun plani takipcisine bagli degil -- nuPlan
#   CLS at-fault carpismayi cezalandiriyor, arkadan carpilmayi degil. Olculdu
#   (500 sahne, ajanin ego'ya gore koridor yayi uzerindeki boylamsal konumu):
#     behind  n=228   8 s SONUNDA ego'nun onunde %2.6   ufukta bir an onune gecen %2.6
#     ahead   n=219   8 s SONUNDA ego'nun onunde %98.6  ufukta bir an onune gecen %100.0
#   Takipcilerin %97.4'u ufuk boyunca arkada kaliyor -> ongorulecek etkilesim yok,
#   ve L1'in hicbir sinifina (yield/wait/merge/overtake) donusemiyor; hepsi diger
#   ajanin onde veya kesisen olmasini gerektiriyor.
#   Serit degistirirken hedef seritteki takipci ONEMLI, ama o leftAdjacent /
#   rightAdjacent'in isi -- bizde lon_ok = ds >= -BEHIND_MAX_M ile arkadakiler
#   zaten dahil. same_lane_behind spesifik olarak KENDI seridimizdeki takipci.
#
# GERI GETIRME SECENEGI: NEGATIF KONTROL olarak. Mudahale ettigimizde plani
#   DEGISTIRMEMESI gereken bir kavram, intervention-correctness denetimi icin
#   capa olurdu (o denetimi su an -9.0 puanla kaybediyoruz). Bir slota mal olur.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# KAPATILDI: onRouteCorridor  (KG reason "ego_route_corridor", agirlik 40;
#            nuplan_predicates_modular/categories/map.py:775 -- ajanin roadblock'u
#            ego'nun rota roadblock listesindeyse dogru)
#
# NEDEN: kavram kumesinde SUBSUMPTION yapiyordu. Olculdu (1118 sahne validation):
#     P(onRouteCorridor | same_lane_ahead)      = 93.6%
#     P(onRouteCorridor | staticObstacleOnPath) = 94.1%
#     P(onRouteCorridor | leftAdjacent)         = 91.3%
#     P(onRouteCorridor | same_lane_behind)     = 90.0%
#   yani dort serit kanalini birden iceriyordu (%39.6 yanma). CBM'de bir kanal
#   digerlerini kapsayinca atif belirsizlesiyor: karar hangi kavramdan geldi
#   ayirt edilemiyor. Koridoru daraltmak COZMUYOR -- hangi koridor tanimi olursa
#   olsun ego seridi icinde kalir, dolayisiyla same_lane_ahead her zaman altkumesi
#   olur; KG'nin dar varyanti generator_forward_corridor (|lat| <= 7.0 m, agirlik
#   105) ise +/-2 serit genisliginde oldugu icin adjacent'lari da kapsar.
#
# NE KAYBEDIYORUZ: bu kanalin TEK basina getirdigi ajanlar cop DEGIL. Olculdu
#   (500 sahne, ajanin 8 s GT ufkunda ego'ya en yakin merkez-merkez mesafesi):
#     TEK onRouteCorridor    n=536   medyan 10.3 m   %95.1'i 20 m'ye giriyor
#     baska kanal yanan      n=2236  medyan  9.2 m   %88.5'i 20 m'ye giriyor
#     hicbir kanal yanmayan  n=1898  medyan 15.2 m   %63.6'si 20 m'ye giriyor
#   Yani hicbir serit iliskisiyle aciklanamayan ama gercekten yaklasan ajanlar.
#   Kapali oldugu surece bu ~%11 ajan grafige HIC girmiyor.
#
# GERI GETIRME SECENEKLERI (plan netlesince karar verilecek):
#   (a) RESIDUAL kural: ajan icin baska hicbir kanal yanmiyorsa yansin. Overlap
#       insaat geregi 0 olur, kapsama aynen korunur, yanma %39.6 -> ~%11.
#   (b) L1'e tasi: "rotada ama yapisal iliskiyle aciklanamayan" bir ANTICIPATORY
#       kavram olarak L1'de degerlendir (L0 yapisal seciciligi bozmadan).
# Kod asagida compute_agent_channels icinde yorumda duruyor; silinmedi.
# --------------------------------------------------------------------------

# ---------------- L0 HARITA (7) -- 2026-09-03 yeniden tasarim ----------------
# Iki totolojik kanal KALDIRILDI:
#   inCrosswalk    = is_cw & valid  -> elemanin TURUNU tekrar ediyordu (model token
#                    pozisyonundan zaten biliyor); yerine crossesEgoPath (asagida).
#   onExpertRoute  = is_rt & reach  -> ayni sekilde tur tekrari; kaldirildi, yerine
#                    kanal konmadi (ego'nun gececegi yol zaten inLane/successor).
# Bir kanal EKLENDI:
#   crossesEgoPath = elemanin geometrisi ego koridorunu KESIYOR (ajan tarafindaki
#                    kesisim filtresiyle ayni isaret-degisimi testi). Yolu kesen
#                    GECIT de kesen SERIT de burada yanar -- tur kimligi degil,
#                    ego planiyla yapisal cakisma bilgisi tasir.
# Iki kanal YENIDEN TANIMLANDI (asil sorunlar):
#   inIntersection = eleman, EGO'NUN koridorunun girdigi kavsak poligonundan
#                    geciyor (once: elemanin en-yakin noktasi HERHANGI bir
#                    poligondaysa tum eleman yaniyordu -> %22.6).
#   trafficControl = eleman EGO'NUN yolunun ustunde VE (gercek TL tasiyor VEYA
#                    dur poligonuna degiyor) (once: "herhangi bir TL kaydi var"
#                    -> capraz trafigin isikli seritleri de yaniyordu -> %24.1).
M_IN_LANE = 0
M_LEFT_ADJACENT = 1
M_RIGHT_ADJACENT = 2
M_SUCCESSOR = 3
M_IN_INTERSECTION = 4
M_CROSSES_EGO_PATH = 5
M_TRAFFIC_CONTROL = 6
NUM_M = 7
M_NAMES = ["inLane", "leftAdjacent", "rightAdjacent", "successor",
           "inIntersection", "crossesEgoPath", "trafficControl"]

# --- esikler (KG relevance_logic.py'den; sapma olanlar isaretli) ---
AHEAD_MAX_M = 80.0        # KG directional_same_lane_ahead_m (eski bizim degerimiz 40 idi)
BEHIND_MAX_M = 25.0       # KG directional_same_lane_behind_m
ADJ_MAX_M = 35.0          # KG directional_adjacent_m (40 idi; kalibrasyonda hizalandi)
VRU_MAX_M = 35.0          # KG vulnerable_road_user_distance_m (12 idi; kalibrasyonda hizalandi)
SHARED_INT_MAX_M = 55.0   # KG shared_intersection_distance_m (relevance_logic.py:43)
# Boylamsal menzil kalibrasyonu (kullanici karari, 2026-09-03): one dogru menzil
# EGO HIZINA bagli -- dururken 60 m ilerideki arac ilgisiz, 15 m/s'de 40 m'deki
# ilgili. ds_max = clamp(v_ego * LON_TAU_S, LON_MIN_M, AHEAD_MAX_M); KG'nin statik
# 80/35 tavanlari DIS sinir olarak korunur. Arkaya dogru menzil sabit LON_BEHIND_M
# (eski -25 m cok genisti; hedef-serit takipcisi icin yakin olan yeter).
LON_TAU_S = 5.0           # * one dogru zaman araligi [s] (3 denendi; 5'e cikarildi)
LON_MIN_M = 25.0          # * one dogru taban; 10 cok dusuktu (validation'da ego p50 2.2 m/s,
                          #   liderler p50 15 m'de -> %74 kayip). Supurme: 10->%74, 20->%23,
                          #   25->%14 kayip. Durgun kuyrukta 3-4 arac boyu lideri korur.
LON_BEHIND_M = 10.0       # * arkaya dogru sabit menzil (eski BEHIND_MAX_M=25 yerine)
EGO_L, EGO_W = 4.62, 2.10 # ego ayak izi [m]
STATIC_SPEED = 0.5        # * duran nesne esigi [m/s]
STATIC_MAX_M = 60.0       # * yolumuzda "ilgili" sayilacak menzil
ROUTE_LAT_TOL = 1.75      # yarim serit
CROSSWALK_TOL = 2.0       # * ajanin gecide yakinlik esigi
CW_LOOKAHEAD_M = 50.0     # * gecidin koridorda ileride olmasi gereken menzil
TL_STOP_TOL = 3.0         # * dur cizgisi / kontrol elemanina yakinlik


def _poly_valid(poly):
    """poly [B,K,P,3] -> nokta gecerli [B,K,P], eleman gecerli [B,K]"""
    pv = poly[..., :2].abs().sum(-1) > 1e-6
    return pv, pv.any(-1)


def _point_in_polys(pts, poly, poly_pt_valid, poly_valid):
    """pts [B,N,2] her poligonun icinde mi -> [B,N,K] bool. Ray casting, ego-frame.

    Poligonlar [K,P,3] sabit P=20 ile SONDAN sifir-dolgulu gelir; gercek nokta sayisi
    V<=P. Halkanin (V-1 -> 0) kapanis kenari, dolgu noktalari ATILARAK kaybediliyordu
    ve ray casting acik polyline uzerinde calisiyordu -> parite yanlis, uzaktaki
    noktalar "icerde" cikiyordu. (Olculdu: intersections'in %26'si, stop_polygons'un
    %100'u V<P.) Cozum: gecersiz noktalari ILK noktaya cokert. Boylece kapanis kenari
    (V-1 -> ilk) geri gelir ve kalan dolgu kenarlari sifir uzunluklu olur; sifir
    uzunluklu kenarda y1 == y2 oldugu icin kesisim sayaci onlari zaten saymaz.
    """
    B, N, _ = pts.shape
    K, P = poly.shape[1], poly.shape[2]
    first = poly[..., :1, :2]                                 # [B,K,1,2] (ilk nokta hep gecerli)
    pxy = torch.where(poly_pt_valid.unsqueeze(-1), poly[..., :2], first.expand(-1, -1, P, -1))
    px, py = pxy[..., 0], pxy[..., 1]                         # [B,K,P]
    nxt = torch.roll(px, -1, dims=2), torch.roll(py, -1, dims=2)
    x = pts[:, :, None, None, 0]                              # [B,N,1,1]
    y = pts[:, :, None, None, 1]
    x1, y1 = px[:, None], py[:, None]                         # [B,1,K,P]
    x2, y2 = nxt[0][:, None], nxt[1][:, None]
    cond = ((y1 > y) != (y2 > y))
    denom = (y2 - y1)
    denom = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    xint = x1 + (y - y1) * (x2 - x1) / denom
    crossings = (cond & (x < xint)).sum(-1)                   # [B,N,K]
    return (crossings % 2 == 1) & poly_valid[:, None]



def _footprint_pts(pos, theta, length, width):
    """Varlik ayak izini 5 nokta ile temsil et: merkez + 4 kose. [B,N,5,2].

    KG, kavsak uyeligini IKI kumede tutuyor (categories/map.py:505-535):
      intersection_ids           <- center_covered      (merkez poligonun icinde)
      intersection_footprint_ids <- footprint overlap   (kutu poligonla ortusuyor)
    ve ciftte bu iki kumenin BIRLESIMINI kesistiriyor. Tam kutu-poligon kesisimi
    yerine kose noktalarini test etmek bu birlesimin yakin bir karsiligi: merkez
    icerdeyse birinci kume, herhangi bir kose icerdeyse ikinci kume yakalanir.
    """
    c, sn = torch.cos(theta), torch.sin(theta)
    hl, hw = 0.5 * length, 0.5 * width
    # yerel kose ofsetleri (+-hl, +-hw) -> global
    ox = torch.stack([torch.zeros_like(hl), hl, hl, -hl, -hl], dim=-1)      # [B,N,5]
    oy = torch.stack([torch.zeros_like(hw), hw, -hw, hw, -hw], dim=-1)
    gx = pos[..., 0:1] + ox * c.unsqueeze(-1) - oy * sn.unsqueeze(-1)
    gy = pos[..., 1:2] + ox * sn.unsqueeze(-1) + oy * c.unsqueeze(-1)
    return torch.stack([gx, gy], dim=-1)                                    # [B,N,5,2]


def _entity_in_polys(fp, poly, ppv, pv):
    """fp [B,N,5,2] -> [B,N,K]: varligin HERHANGI bir noktasi poligonun icinde mi."""
    B, N, P5, _ = fp.shape
    hit = _point_in_polys(fp.reshape(B, N * P5, 2), poly, ppv, pv)          # [B,N*5,K]
    return hit.view(B, N, P5, -1).any(2)


def compute_agent_channels(neighbor_agents_past, ego_agent_past, ref_path,
                           route_lanes, crosswalks, intersections, stop_polygons,
                           lane_tl, map_lanes, neighbor_valid=None):
    """L0 v3 ajan kanallari. GELECEK KULLANMAZ -- hepsi anlik/yapisal (Kisit 1).
    Doner active [B,N,10] bool."""
    B, N = neighbor_agents_past.shape[:2]
    dev = neighbor_agents_past.device
    cur = neighbor_agents_past[:, :, -1]
    pos, theta, vel = cur[..., 0:2], cur[..., 2], cur[..., 3:5]
    if neighbor_valid is None:
        neighbor_valid = cur[..., :2].abs().sum(-1) > 1e-6
    tip = cur[..., 8:11].argmax(-1)                            # 0=veh 1=ped 2=bic
    veh_like = tip != 1
    vru = tip != 0                                             # yaya + bisiklet
    speed = vel.norm(dim=-1)

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
    s_j, d_lat, tan_yaw, on_start = _project(pos, cxy, cyaw, ccum, cvalid)
    behind = on_start & (pos[..., 0] < 0)
    ds = torch.where(behind, pos[..., 0], s_j)
    d_lat_eff = torch.where(behind, pos[..., 1], d_lat)
    tan_eff = torch.where(behind, torch.zeros_like(tan_yaw), tan_yaw)
    same_flow = _wrap(theta - tan_eff).abs() <= SAME_FLOW_RAD
    d_center = pos.norm(dim=-1)

    inlane = d_lat_eff.abs() <= 0.5 * LANE_W
    adjL = (d_lat_eff > 0.5 * LANE_W) & (d_lat_eff <= 1.5 * LANE_W)
    adjR = (d_lat_eff < -0.5 * LANE_W) & (d_lat_eff >= -1.5 * LANE_W)

    act = torch.zeros(B, N, NUM_A, dtype=torch.bool, device=dev)
    v_ego = ego_agent_past[:, -1, 3:5].norm(dim=-1)                        # [B]
    lon_ahead = (v_ego * LON_TAU_S).clamp(min=LON_MIN_M, max=AHEAD_MAX_M).unsqueeze(-1)
    act[..., A_SAME_LANE_AHEAD] = inlane & same_flow & (ds > 0) & (ds <= lon_ahead) & veh_like
    act[..., A_SAME_LANE_BEHIND] = (inlane & same_flow & (ds < 0)
                                    & (ds >= -LON_BEHIND_M) & veh_like)
    lon_ok = (ds >= -LON_BEHIND_M) & (ds <= lon_ahead)
    act[..., A_LEFT_ADJACENT] = adjL & same_flow & (d_center <= ADJ_MAX_M) & lon_ok & veh_like
    act[..., A_RIGHT_ADJACENT] = adjR & same_flow & (d_center <= ADJ_MAX_M) & lon_ok & veh_like

    # sharesIntersectionWith -- KG tasarimina sadik (categories/map.py:926-939 +
    # relevance_logic.py:248-250):
    #   subject_ids = intersection_ids U intersection_footprint_ids   (ego)
    #   object_ids  = ayni                                            (ajan)
    #   shared      = bool(subject_ids & object_ids)
    #   kapi        = center <= shared_intersection_distance_m = 55 m
    # ONCEKI (yanlis) hal: ego icin KORIDORUN 0-60 m ilerisi kullaniliyordu. KG
    # ego'nun SU ANKI ayak izini istiyor -- ego henuz kavsaga varmadan yanmamali.
    # Ayrica ajan icin sadece merkez noktasi test ediliyordu; KG ayak izi
    # ortusmesini de sayiyor.
    ipv, iv = _poly_valid(intersections)
    ego_fp = _footprint_pts(torch.zeros(B, 1, 2, device=dev),
                            torch.zeros(B, 1, device=dev),
                            torch.full((B, 1), EGO_L, device=dev),
                            torch.full((B, 1), EGO_W, device=dev))               # [B,1,5,2]
    ego_in = _entity_in_polys(ego_fp, intersections, ipv, iv)[:, 0]              # [B,I]
    ag_fp = _footprint_pts(pos, theta,
                           cur[..., 6].clamp(min=1.0), cur[..., 7].clamp(min=0.6))
    agent_in = _entity_in_polys(ag_fp, intersections, ipv, iv)                   # [B,N,I]
    kg_shared = (agent_in & ego_in[:, None]).any(-1) & (d_center <= SHARED_INT_MAX_M)

    # --- KESISIM FILTRESI (bizim karar-ilgisi kisitimiz, KG'de YOK) ---------------
    # !!! BILINEN SORUN (2026-09-03, cozum ertelendi): kesisme noktasi s ∈ [-5, 80]
    # kabul ediliyor. Ego catisma noktasini GECTIKTEN sonra (kavsaktan cikis fazi,
    # ego kutuya hala degiyorken) ARKADAKI ajan yanabiliyor -- ajan ego'nun
    # gelecegini degil gecmisini kesiyor (olculdu: e990429a8ff45085, ajan 4,
    # (-19.5, -0.1), poligon merkezi ego'nun 8 m arkasinda). Aday duzeltme:
    # flips kosulundaki (ls > -5.0) -> (ls > 0.0). Karar verilmeden degistirme.
    # KG'nin testi salt uyelik oldugu icin ayni kavsak kutusundaki AYNI-YON ajanlar
    # da yaniyordu (%69.9; ajan-bazinda KG ile 8/8 dogrulandi -- yani KG'nin kendisi
    # de boyle). Adindaki "conflict"i gercek yapan kisit: ajanin SERIDI ego
    # koridorunu KESMELI. Yapisal -- gelecek yok, hiz yok; heading yalnizca serit
    # secimi ve kesisme acisi icin kullanilir (L0 Kisit 1 korunur).
    #
    # 1) Serit-kesisme: serit polyline'inin koridora gore yanal isareti ardil iki
    #    noktada degisiyorsa (merkez cizgiyi geciyor) VE o noktadaki serit yonu
    #    koridor tegetinden SAME_FLOW_RAD'dan fazla sapiyorsa -> kesen serit.
    #    (Aci sarti, ego'nun kendi seridinin lat~0 titremesini eler.)
    lxy = map_lanes[..., :2]                                        # [B,L,P,2]
    lpv = lxy.abs().sum(-1) > 1e-6
    L_, P_ = lxy.shape[1], lxy.shape[2]
    ls, llat, ltan, _ = _project(lxy.reshape(B, -1, 2), cxy, cyaw, ccum, cvalid)
    ls, llat, ltan = ls.view(B, L_, P_), llat.view(B, L_, P_), ltan.view(B, L_, P_)
    lhd = map_lanes[..., 2]
    sgn = torch.sign(llat)
    pairv = lpv[:, :, :-1] & lpv[:, :, 1:]
    flips = (pairv & (sgn[:, :, :-1] * sgn[:, :, 1:] < 0)
             & (ls[:, :, :-1] > -5.0) & (ls[:, :, :-1] <= AHEAD_MAX_M)
             & (_wrap(lhd - ltan)[:, :, :-1].abs() > SAME_FLOW_RAD))
    lane_crosses = flips.any(-1)                                    # [B,L]

    # 2) Ajan -> serit atamasi: en yakin serit (nokta mesafesi <= 3 m) icinde,
    #    ajan yonuyle en uyumlu olani sec.
    d_al = torch.cdist(pos, lxy.reshape(B, -1, 2)).masked_fill(
        ~lpv.reshape(B, 1, -1), 1e9).view(B, N, L_, P_)             # [B,N,L,P]
    dmin, imin = d_al.min(-1)                                       # [B,N,L]
    hd_at = torch.gather(lhd.unsqueeze(1).expand(-1, N, -1, -1), 3,
                         imin.unsqueeze(-1)).squeeze(-1)            # [B,N,L]
    align = _wrap(theta.unsqueeze(-1) - hd_at).abs() <= math.radians(60.0)
    cost = dmin + (~align).float() * 1e6 + (dmin > 3.0).float() * 1e6
    best_cost, best_lane = cost.min(-1)                             # [B,N]
    has_lane = best_cost < 1e6
    my_lane_crosses = torch.gather(lane_crosses.unsqueeze(1).expand(-1, N, -1),
                                   2, best_lane.unsqueeze(-1)).squeeze(-1)

    # 3) Serit atanamayanlarda (baglantisiz/kavsak ici sapmis ajan) geri dusus:
    #    ajanin ANLIK yonu koridor tegetini kesiyor mu (30-150 derece).
    dth_c = _wrap(theta - tan_eff).abs()
    heading_crossing = (dth_c > SAME_FLOW_RAD) & (dth_c < math.pi - SAME_FLOW_RAD)
    conflict = torch.where(has_lane, my_lane_crosses, heading_crossing)

    act[..., A_SHARES_INTERSECTION] = kg_shared & conflict & neighbor_valid

    # VRU: yaya/bisiklet, koridora yakin, menzil icinde
    act[..., A_VRU_NEAR_PATH] = (vru & (d_center <= VRU_MAX_M)
                                 & (d_lat_eff.abs() <= 1.5 * LANE_W) & (ds >= -5.0))

    # inCrosswalk -- KAPATILDI (gerekce icin dosya basindaki blok). Geri getirilirse
    # ESKI kod DEGIL, asagidaki DOGRU tanim kullanilmali (gecit-bazli konjonksiyon):
    #   cpv, cvv_ = _poly_valid(crosswalks)
    #   inside_k = _point_in_polys(pos, crosswalks, cpv, cvv_)               # [B,N,C]
    #   cs, cl, _, _ = _project(crosswalks[..., :2].reshape(B, -1, 2), cxy, cyaw, ccum, cvalid)
    #   on_path_k = ((cl.abs() <= LANE_W) & (cs > 0) & (cs <= CW_LOOKAHEAD_M)
    #                ).view(B, C, P).any(-1)                                 # [B,C] -- SAHNE degil
    #   act[..., A_IN_CROSSWALK] = (inside_k & on_path_k[:, None]).any(-1) & neighbor_valid

    # staticObstacleOnPath -- KAPATILDI (gerekce icin dosya basindaki blok). Kod korundu:
    #   act[..., A_STATIC_OBSTACLE] = (inlane & (ds > 0) & (ds <= STATIC_MAX_M)
    #                                  & (speed <= STATIC_SPEED) & neighbor_valid)

    # onRouteCorridor -- KAPATILDI (gerekce icin dosya basindaki blok). Kod korundu:
    #   rl = route_lanes[..., :2]
    #   rl_v = rl.abs().sum(-1) > 1e-6
    #   d_rl = torch.cdist(pos, rl.reshape(B, -1, 2)).masked_fill(~rl_v.reshape(B, 1, -1), 1e9)
    #   on_route = ((d_rl.min(-1).values <= ROUTE_LAT_TOL)
    #               & (ds <= AHEAD_MAX_M) & (ds >= -BEHIND_MAX_M) & neighbor_valid)
    #   # (a) residual olarak geri getirmek icin: act[..., A_ON_ROUTE_CORRIDOR] = (
    #   #        on_route & ~act.any(-1))   <-- diger tum kanallar hesaplandiktan SONRA

    # sharedTrafficControl -- KAPATILDI (gerekce icin dosya basindaki blok).
    # Eski (yanlis) kod, "shared" sarti olmadan:
    #   tl_real = lane_tl[..., :3].abs().sum(-1) > 1e-6
    #   ctrl_pts = torch.where(tl_real.unsqueeze(-1), map_lanes[..., :2], 0)
    #   stop_pts = stop_polygons[..., :2] (gecerli olanlar)
    #   d_ctrl = cdist(pos, cat([ctrl_pts, stop_pts]))
    #   act[..., A_SHARED_TRAFFIC_CONTROL] = (d_ctrl.min(-1).values <= TL_STOP_TOL)
    #                                        & (ds <= AHEAD_MAX_M) & neighbor_valid

    return act & neighbor_valid.unsqueeze(-1)


def compute_map_channels_v3(map_lanes, map_crosswalks, route_lanes, ref_path,
                            lane_tl, intersections, stop_polygons):
    """L0 v3 harita kanallari. Doner active [B,S,8] bool (S = L + C + R, model sirasi)."""
    B, L, P, _ = map_lanes.shape
    dev = map_lanes.device
    C, R = map_crosswalks.shape[1], route_lanes.shape[1]
    Pm = max(P, map_crosswalks.shape[2], route_lanes.shape[2])

    def pad(t):
        if t.shape[2] == Pm:
            return t
        z = torch.zeros(t.shape[0], t.shape[1], Pm - t.shape[2], t.shape[3],
                        dtype=t.dtype, device=t.device)
        return torch.cat([t, z], dim=2)

    exy = torch.cat([pad(map_lanes[..., :3]), pad(map_crosswalks[..., :3]),
                     pad(route_lanes[..., :3])], dim=1)                      # [B,S,Pm,3]
    S = exy.shape[1]
    ehd = exy[..., 2]
    xy = exy[..., :2]
    pv = xy.abs().sum(-1) > 1e-6
    elem_valid = pv.any(-1)
    is_cw = torch.zeros(B, S, dtype=torch.bool, device=dev); is_cw[:, L:L + C] = True
    is_rt = torch.zeros(B, S, dtype=torch.bool, device=dev); is_rt[:, L + C:] = True

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
    d_ego = xy.norm(dim=-1).masked_fill(~pv, 1e9)
    min_d, min_i = d_ego.min(-1)
    hd_near = torch.gather(ehd, 2, min_i.unsqueeze(-1)).squeeze(-1)
    fs, fdlat, ftan, _ = _project(xy.reshape(B, S * Pm, 2), cxy, cyaw, ccum, cvalid)
    fs, fdlat = fs.view(B, S, Pm), fdlat.view(B, S, Pm)
    tan_near = torch.gather(ftan.view(B, S, Pm), 2, min_i.unsqueeze(-1)).squeeze(-1)

    act = torch.zeros(B, S, NUM_M, dtype=torch.bool, device=dev)
    aligned = torch.cos(hd_near) > math.cos(MAP_DIR_RAD)
    act[..., M_IN_LANE] = (min_d <= 0.5 * LANE_W) & aligned & ~is_cw

    med = torch.where(pv, fdlat, torch.zeros_like(fdlat)).sum(-1) / pv.float().sum(-1).clamp(min=1)
    par = torch.cos(hd_near - tan_near) > math.cos(MAP_DIR_RAD)
    reach = torch.where(pv, xy[..., 0], torch.full_like(xy[..., 0], -1e9)).amax(-1) > 0
    act[..., M_LEFT_ADJACENT] = (med > 0.5 * LANE_W) & (med <= 1.5 * LANE_W) & par & reach & ~is_cw
    act[..., M_RIGHT_ADJACENT] = (med < -0.5 * LANE_W) & (med >= -1.5 * LANE_W) & par & reach & ~is_cw

    on_corr = (fdlat.abs() <= 0.5 * LANE_W) & (fs > 1.0)
    frac = (on_corr & pv).float().sum(-1) / pv.float().sum(-1).clamp(min=1)
    act[..., M_SUCCESSOR] = (frac > 0.3) & ~act[..., M_IN_LANE] & ~is_cw

    # inIntersection (yeniden tanim): eleman, EGO'NUN kavsagindan geciyor.
    #   ego'nun kavsagi = koridorun 0-80 m'de girdigi poligon(lar);
    #   eleman uyeligi = HERHANGI bir eleman noktasi o poligonun icinde
    #   (once: elemanin sadece ego'ya en yakin noktasi, HERHANGI bir poligonda).
    ipv, iv = _poly_valid(intersections)
    ego_pts = _arc_walk(cxy, ccum, cvalid,
                        torch.linspace(0, 80, 17, device=dev)[None].expand(B, -1))
    ego_ix = _point_in_polys(ego_pts, intersections, ipv, iv).any(1)         # [B,I]
    el_in = _point_in_polys(xy.reshape(B, S * Pm, 2), intersections, ipv, iv)
    el_in = (el_in.view(B, S, Pm, -1) & pv.unsqueeze(-1)).any(2)             # [B,S,I]
    act[..., M_IN_INTERSECTION] = (el_in & ego_ix[:, None]).any(-1) & elem_valid

    # crossesEgoPath (YENI): elemanin polyline'i ego koridorunun merkez cizgisini
    # kesiyor -- ajan tarafindaki kesisim filtresiyle ayni test: ardil gecerli iki
    # noktada yanal isaret degisimi, kesisme ILERIDE (0 < s <= 80) ve o noktadaki
    # eleman yonu koridor tegetinden SAME_FLOW_RAD'dan fazla sapiyor.
    sgn = torch.sign(fdlat)
    pairv_m = pv[:, :, :-1] & pv[:, :, 1:]
    ang = _wrap(ehd - ftan.view(B, S, Pm))[:, :, :-1].abs()
    mflips = (pairv_m & (sgn[:, :, :-1] * sgn[:, :, 1:] < 0)
              & (fs[:, :, :-1] > 0.0) & (fs[:, :, :-1] <= AHEAD_MAX_M)
              & (ang > SAME_FLOW_RAD))
    act[..., M_CROSSES_EGO_PATH] = mflips.any(-1) & elem_valid & ~act[..., M_IN_LANE]

    # trafficControl (yeniden tanim): EGO'NUN YOLUNUN kontrollu parcasi.
    #   eleman ego koridorunun ustunde ILERIDE (|lat| <= yarim serit, 0 < s <= 80)
    #   VE (gercek TL durumu tasiyor VEYA dur poligonuna TL_STOP_TOL icinde degiyor).
    on_ego_path = ((fdlat.abs() <= 0.5 * LANE_W) & (fs > 0.0)
                   & (fs <= AHEAD_MAX_M) & pv).any(-1)                       # [B,S]
    tl_real = torch.zeros(B, S, dtype=torch.bool, device=dev)
    tl_real[:, :L] = (lane_tl[..., :3].abs().sum(-1) > 1e-6).any(-1)
    spv, sv = _poly_valid(stop_polygons)
    sp = stop_polygons[..., :2].reshape(B, -1, 2)
    sp_v = spv.reshape(B, -1)
    d_sp = torch.cdist(xy.reshape(B, S * Pm, 2), sp).masked_fill(~sp_v[:, None], 1e9)
    d_sp = d_sp.min(-1).values.view(B, S, Pm)
    touches_stop = d_sp.masked_fill(~pv, 1e9).min(-1).values <= TL_STOP_TOL
    act[..., M_TRAFFIC_CONTROL] = on_ego_path & (tl_real | touches_stop) & elem_valid

    return act & elem_valid.unsqueeze(-1)
