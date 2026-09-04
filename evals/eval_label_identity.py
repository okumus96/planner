"""LABEL IDENTITY TEST — predicate ETIKET KIMLIGI karari etkiliyor mu?

Soru: bir (ajan, iliski) girdisinin ETIKETI degistirildiginde -- ajanin dugum ozelligi,
kenar geometrisi, evidence degerleri ve sahnenin geri kalani BIT-BIT ayni kalirken --
ilan edilen karar b* degisiyor mu?

Tek duzenlenen tensor: inputs["channel_active"][b, j, :] (11 bitlik bool satir).

DURUMLAR (hedef ajan j icin; ayni batarya kontrol ajanina da uygulanir):
  base            : dokunulmamis sahne
  only_<r>        : maske = TAM OLARAK {r}, r in REL_TEST (8 iliski)
  cut             : maske = {} -> gate ajani causal graftan atar

Herhangi bir (r_base, r_swap) cifti POST-HOC bir kontrasttir:
  S1 = only_<r_base>,  S2 = only_<r_swap>,  S3 = cut
  S1 vs S2 = SAF ETIKET etkisi (girdi sayisi 1 -> 1, eslesmis A/B)
  S1 vs S3 = AJANIN VARLIGI etkisi (girdi sayisi 1 -> 0, doz tavani / filtre dayanagi)

Bu yuzden 3 durum x 9 cift degil, |REL_TEST|+1 ileri gecis yeter (+base).

HEDEF SECIMI (mekanik, elle kurasyon yok): ahead-tipi ({follows, same_lane_ahead})
en az bir iliski yakan komsular icinde M_cas_typed kutlesi EN YUKSEK olan.
Boyle komsu yoksa sahne atlanir ve sayilir.
KONTROL: ahead-tipi hicbir iliski yakmayan EN UZAK gecerli komsu (gurultu tabani).

METRIKLER (durum basina kaydedilir, kontrastlar analizde):
  P4 = 4-sinif lon dagilimi (LON4: stop/slow/accel/maintain); 9x7 modeller LON4_MAP ile katlanir
  P5 = 5-sinif lat dagilimi (LAT5V); 9x7 modeller LAT5V_MAP ile katlanir
  p_slow = P4[stop]+P4[slow] (eski dP_SLOW ile uyumlu), p_slow_only = P4[slow], p_go = P4[accel]+P4[maintain]
  dv = plan uc-hiz farki (base'e gore), dplan = plan L2 (base'e gore)
  TESHIS: dikkat doygunlugu (tepe kutle, normalize entropi) + hedefin kutle payi

Bu script IKI DALDA da calisir: CausalPlanner ctor'una hangi kwarg'lar varsa onlar gecirilir
(channels: lat_moe; relation-bottleneck: rel_bottleneck/rel_evidence).

Kosum (v2, channels dali):
  python evals/eval_label_identity.py \
    --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/v3_latmoe/causal_epoch_12_minADE_0.7998.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    --tag v2 --lat_moe 1 --device cuda:0
"""
import argparse
import inspect
import json
import os
import time

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# evals/ altindan cagrilsa da repo kokunu import yoluna al (iki dalda da ayni davransin)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.channels import (CHANNEL_NAMES, NUM_CHANNELS,
                                 CH_FOLLOWS, CH_SAME_LANE_AHEAD, CH_SAME_LANE_BEHIND,
                                 CH_ADJACENT_LEFT, CH_ADJACENT_RIGHT, CH_COLLISION_COURSE,
                                 CH_SHARES_INTERSECTION, CH_NEAR, CH_MERGES, CH_VRU,
                                 EV_TTC, EV_CLOSING, EV_DS, EV_DFS, MCH_TRAFFIC)
from GameFormer.decision_labels import LON_CLASSES
try:
    # channels dali: 4x5 sozluk tablolari orada tanimli
    from GameFormer.decision_labels import (LON4_MAP, LON4_CLASSES, LAT5V_MAP, LAT5V_CLASSES,
                                            NUM_LON4, NUM_LAT5V)
except ImportError:
    # relation-bottleneck dali 4x5 sozlugu ONCELER -> tablolari burada birebir tekrarla
    # (kaynak: channels@683b6c4 GameFormer/decision_labels.py:97-103). Katlama tanimi
    # iki dalda AYNI olmali, yoksa modeller karsilastirilamaz.
    LON4_MAP = [0, 0, 0, 1, 1, 2, 2, 3, 0]        # LON_CLASSES(9) -> LON4
    LON4_CLASSES = ['stop', 'slow', 'accel', 'maintain']
    LAT5V_MAP = [0, 1, 2, 3, 2, 3, 4]             # LAT_CLASSES(7) -> LAT5V
    LAT5V_CLASSES = ['turn_left', 'turn_right', 'to_left', 'to_right', 'none']
    NUM_LON4, NUM_LAT5V = len(LON4_CLASSES), len(LAT5V_CLASSES)
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

# --- hedef secimi: "ahead-tipi" iliskiler (simdiki kinematikten, tahminsiz, isaret geregi onde) ---
AHEAD = [CH_FOLLOWS, CH_SAME_LANE_AHEAD]
CAUTION = AHEAD + [CH_COLLISION_COURSE, CH_SHARES_INTERSECTION, CH_MERGES, CH_VRU]

# --- tek-iliski bataryasi (kullanici onayi): 3 r_base adayi + benign/caution r_swap'ler +
#     same_lane_behind (geometri "onumde" derken etiket "arkamda" -- en sert kimlik testi) ---
REL_TEST = [CH_SAME_LANE_AHEAD, CH_FOLLOWS, CH_COLLISION_COURSE, CH_SHARES_INTERSECTION,
            CH_NEAR, CH_ADJACENT_RIGHT, CH_ADJACENT_LEFT, CH_SAME_LANE_BEHIND]

SLOW_GT = [LON_CLASSES.index(c) for c in
           ('stop_quickly', 'stop_gently', 'slow_quickly', 'slow_gently')]
URGENT_GT = [LON_CLASSES.index(c) for c in ('stop_quickly', 'slow_quickly')]


def state_names():
    return (['base']
            + [f'tgt_only_{CHANNEL_NAMES[r]}' for r in REL_TEST] + ['tgt_cut']
            + [f'ctl_only_{CHANNEL_NAMES[r]}' for r in REL_TEST] + ['ctl_cut'])


def build_model(a, dev):
    """Iki dalda da calisan ctor: yalnizca IMZADA VAR OLAN kwarg'lar gecirilir."""
    kw = dict(layers=a.graph_layers, modes=a.modes, nbr_enrich=a.nbr_enrich,
              ego_residual=a.ego_residual, gate_channels=1, typed_kv=a.typed_kv,
              dod_meta=a.dod_meta)
    sig = set(inspect.signature(CausalPlanner.__init__).parameters)
    have_latmoe = 'lat_moe' in sig
    if a.lat_moe:
        if not have_latmoe:
            raise SystemExit("--lat_moe bu daldaki CausalPlanner'da yok (channels dali gerekir)")
        kw.update(lat_moe=a.lat_moe, num_lon=NUM_LON4, num_lat=NUM_LAT5V)
    else:
        kw.update(num_lon=9, num_lat=7)
    if a.l1:
        kw.update(l1=a.l1, l1_bottleneck=a.l1_bottleneck, num_l1_ag=6, num_l1_mp=2)
    for k in ('rel_bottleneck', 'rel_evidence'):
        v = getattr(a, k)
        if v:
            if k not in sig:
                raise SystemExit(f"--{k} bu daldaki CausalPlanner'da yok "
                                 f"(relation-bottleneck dali gerekir)")
            kw[k] = v
        elif k in sig:
            kw[k] = 0
    m = CausalPlanner(**kw).to(dev)
    miss, unexp = m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    if miss or unexp:
        raise SystemExit(f"[load] bayrak/checkpoint uyusmazligi: missing={len(miss)} "
                         f"unexpected={len(unexp)}\n  missing={list(miss)[:8]}\n"
                         f"  unexpected={list(unexp)[:8]}")
    m.eval()
    return m


def best_plan(out, B):
    traj = out['traj'][:, 0]
    best = out['score'][:, 0].argmax(-1)
    return traj[torch.arange(B, device=traj.device), best][..., :2]


def end_speed(xy):
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, 34:40].mean(1)


@torch.no_grad()
def run_state(m, enc, inp, t1, ns, N, lon_map, lat_map, ch=None):
    """Tek ileri gecis -> katlanmis 4x5 dagilim + plan + dikkat teshisleri."""
    if ch is not None:
        inp = dict(inp)
        inp["channel_active"] = ch
    out = m(enc, inp, num_agents=N + 1, neighbor_futures=t1, neighbor_states=ns)
    B = out['psi_lon_cas'].shape[0]
    dev = out['psi_lon_cas'].device
    P_lon = torch.softmax(out['psi_lon_cas'].float(), -1)
    P_lat = torch.softmax(out['psi_lat_cas'].float(), -1)
    P4 = torch.zeros(B, NUM_LON4, device=dev).index_add_(1, lon_map, P_lon)
    P5 = torch.zeros(B, NUM_LAT5V, device=dev).index_add_(1, lat_map, P_lat)
    plan = best_plan(out, B)
    mt = out['M_cas_typed']                                        # [B,N,R], (s,r) uzerinde toplam 1
    ent_valid = out['gated_valid'][:, :, None] & out['ch_active']  # [B,N,R]
    n_ent = ent_valid.flatten(1).sum(1)
    flat = mt.flatten(1)
    peak = flat.max(1).values
    p = flat.clamp(min=1e-12)
    ent = -(p * p.log()).sum(1) / n_ent.clamp(min=2).float().log()
    return dict(P4=P4, P5=P5, plan=plan, v=end_speed(plan), mt=mt,
                n_ent=n_ent, peak=peak, ent=ent)


@torch.no_grad()
def main(a):
    dev = a.device
    os.makedirs(a.out_dir, exist_ok=True)
    N = a.num_neighbors

    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=N)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev)
    freeze_gameformer(gf)
    m = build_model(a, dev)
    print(f"[model] {a.tag}: {a.causal_path}")
    print(f"[model] lat_moe={a.lat_moe} rel_bottleneck={a.rel_bottleneck} "
          f"rel_evidence={a.rel_evidence} -> psi_lon {m.psi_lon[-1].out_features} sinif")

    if a.lat_moe:
        lon_map = torch.arange(NUM_LON4, device=dev)
        lat_map = torch.arange(NUM_LAT5V, device=dev)
    else:
        lon_map = torch.tensor(LON4_MAP, dtype=torch.long, device=dev)
        lat_map = torch.tensor(LAT5V_MAP, dtype=torch.long, device=dev)

    ds = DrivingData(a.valid_set + "/*.npz", N)
    files = [f.split('/')[-1] for f in ds.data_list]
    idx = list(range(len(ds)))
    if a.limit:
        idx = idx[:a.limit]
    ld = DataLoader(Subset(ds, idx), batch_size=a.batch_size, shuffle=False, num_workers=4)
    print(f"[data] {len(idx)} sahne ({a.valid_set})")

    names = state_names()
    rows = []
    n_batch = 0
    t0 = time.time()
    off = 0
    for batch in ld:
        inp, ef, nf, rp = read_batch(batch, dev)
        B = ef.shape[0]
        ar = torch.arange(B, device=dev)
        enc = gf.encoder(inp)
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, N)

        base = run_state(m, enc, inp, t1, ns, N, lon_map, lat_map)
        ch0 = inp["channel_active"][:, :N].bool()
        nbr_valid = base['mt'].new_ones(B, N, dtype=torch.bool)
        nbr_valid &= inp["neighbor_agents_past"][:, :N].abs().sum((2, 3)) > 0
        ch_v = ch0 & nbr_valid[..., None]

        # --- HEDEF: ahead yakanlar icinde en yuksek causal kutle ---
        ahead = ch_v[..., AHEAD].any(-1)                            # [B,N]
        mass = base['mt'].sum(-1)                                   # [B,N]
        has_tgt = ahead.any(-1)
        tgt = mass.masked_fill(~ahead, -1.0).argmax(-1)             # [B]

        # --- KONTROL: ahead yakmayan EN UZAK gecerli komsu ---
        dist = inp["neighbor_agents_past"][:, :N, -1, :2].norm(dim=-1)
        no_ahead = (~ahead) & nbr_valid
        dist_m = dist.masked_fill(~no_ahead, -1.0)
        ctl = dist_m.argmax(-1)
        has_ctl = dist_m.gather(1, ctl[:, None]).squeeze(1) > 0

        # --- durumlar ---
        outs = {'base': base}
        for who, aidx in (('tgt', tgt), ('ctl', ctl)):
            for r in REL_TEST:
                ch = ch0.clone()
                ch[ar, aidx, :] = False
                ch[ar, aidx, r] = True
                if a.assert_masks:
                    assert not torch.equal(ch, ch0) or bool((ch0[ar, aidx].sum(-1) == 1).all())
                    assert int(ch[ar, aidx].sum(-1).max()) == 1, "only_<r> tek bit olmali"
                    assert bool(ch[ar, aidx, r].all()), "istenen bit yanmali"
                    off_mask = torch.ones(B, N, dtype=torch.bool, device=dev)
                    off_mask[ar, aidx] = False
                    assert torch.equal(ch[off_mask], ch0[off_mask]), "diger ajanlar degismemeli"
                outs[f'{who}_only_{CHANNEL_NAMES[r]}'] = run_state(
                    m, enc, inp, t1, ns, N, lon_map, lat_map, ch=ch)
            ch = ch0.clone()
            ch[ar, aidx, :] = False
            if a.assert_masks:
                assert int(ch[ar, aidx].sum(-1).max()) == 0, "cut tam sifir olmali"
            outs[f'{who}_cut'] = run_state(m, enc, inp, t1, ns, N, lon_map, lat_map, ch=ch)

        # --- sahne kayitlari ---
        evid = inp["channel_evidence"][:, :N]
        mch = inp["map_channel_active"]
        dlon = inp["decision_lon"]
        for i in range(B):
            j, c = int(tgt[i]), int(ctl[i])
            rec = dict(
                scene=idx[off + i], file=files[idx[off + i]],
                has_target=bool(has_tgt[i]), has_ctrl=bool(has_ctl[i]),
                target=j, control=c,
                lon_gt=LON_CLASSES[int(dlon[i])],
                expert_braked=bool(int(dlon[i]) in SLOW_GT),
                expert_urgent=bool(int(dlon[i]) in URGENT_GT),
                traffic_light=bool(mch[i, :, MCH_TRAFFIC].any()),
                n_valid=int(nbr_valid[i].sum()),
                n_ahead=int(ahead[i].sum()),
                n_caution=int((ch_v[i][..., CAUTION].any(-1)).sum()),
                tgt_rels=[CHANNEL_NAMES[r] for r in range(NUM_CHANNELS) if bool(ch_v[i, j, r])],
                ctl_rels=[CHANNEL_NAMES[r] for r in range(NUM_CHANNELS) if bool(ch_v[i, c, r])],
                tgt_ttc=round(float(evid[i, j, EV_TTC]), 3),
                tgt_closing=round(float(evid[i, j, EV_CLOSING]), 3),
                tgt_ds=round(float(evid[i, j, EV_DS]), 2),
                tgt_dfs=round(float(evid[i, j, EV_DFS]), 2),
                ctl_dist=round(float(dist[i, c]), 2),
                base_mass_share=round(float(base['mt'][i, j].sum()
                                            / base['mt'][i].sum().clamp(min=1e-9)), 5),
                states={})
            for nm in names:
                o = outs[nm]
                aidx = j if nm.startswith('ctl') is False else c
                rec['states'][nm] = dict(
                    P4=[round(float(x), 6) for x in o['P4'][i]],
                    P5=[round(float(x), 6) for x in o['P5'][i]],
                    dv=round(float(o['v'][i] - base['v'][i]), 5),
                    dplan=round(float((o['plan'][i] - base['plan'][i]).norm(dim=-1).mean()), 5),
                    mass=round(float(o['mt'][i, aidx].sum()), 6),
                    n_ent=int(o['n_ent'][i]), peak=round(float(o['peak'][i]), 5),
                    ent=round(float(o['ent'][i]), 5))
            rows.append(rec)
        off += B
        n_batch += 1
        if n_batch % 5 == 0:
            el = time.time() - t0
            print(f"  [{off}/{len(idx)}] {el:.0f}s ({el / off * len(idx):.0f}s tahmini)")

    out_path = os.path.join(a.out_dir, f"label_identity_{a.tag}.json")
    json.dump(dict(tag=a.tag, causal_path=a.causal_path,
                   flags=dict(lat_moe=a.lat_moe, rel_bottleneck=a.rel_bottleneck,
                              rel_evidence=a.rel_evidence, graph_layers=a.graph_layers,
                              nbr_enrich=a.nbr_enrich, ego_residual=a.ego_residual,
                              typed_kv=a.typed_kv, dod_meta=a.dod_meta),
                   rel_test=[CHANNEL_NAMES[r] for r in REL_TEST],
                   lon4=LON4_CLASSES, lat5=LAT5V_CLASSES,
                   n_scenes=len(rows), rows=rows),
              open(out_path, "w"))
    print(f"\n[done] {len(rows)} sahne, {time.time() - t0:.0f}s -> {out_path}")
    print(f"  hedef bulundu: {sum(r['has_target'] for r in rows)} / {len(rows)}")
    print(f"  kontrol bulundu: {sum(r['has_ctrl'] for r in rows)} / {len(rows)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out_dir", type=str,
                   default="/home/lt-hta-ai4/GameFormer-Planner/results_label_identity")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--assert_masks", type=int, default=0)
    p.add_argument("--lat_moe", type=int, default=0)
    p.add_argument("--l1", type=int, default=0)
    p.add_argument("--l1_bottleneck", type=int, default=0)
    p.add_argument("--rel_bottleneck", type=int, default=0)
    p.add_argument("--rel_evidence", type=int, default=0)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=2)
    p.add_argument("--typed_kv", type=int, default=1)
    p.add_argument("--dod_meta", type=int, default=1)
    p.add_argument("--ego_residual", type=int, default=0)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda:0")
    main(p.parse_args())
