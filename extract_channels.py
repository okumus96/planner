"""Kanal cikarimi (channels-v1): mevcut islenmis npz cache'ine kanal anahtarlarini ekler.

Eklenen anahtarlar (ilk num_neighbors=10 komsu icin):
  channel_active_gt   [N,11] bool   GT gelecekle ego->agent kanallari (analiz/karsilastirma)
  channel_evidence_gt [N, 9] f32
  channel_active_gf   [N,11] bool   frozen GF top-1 gelecekle (TRAINING BUNU OKUR -- train/deploy tutarliligi)
  channel_evidence_gf [N, 9] f32
  map_channel_active  [S, 8] bool   ego->map kanallari (future kullanmaz, tek surum)
  map_channel_evidence[S, 8] f32
  channels_version    str           surum damgasi (idempotency)

Guvenlik:
  --apply verilmeden HICBIR SEY yazilmaz (kuru kosu: hesaplar, ozetler, dogrular).
  Yazim atomik: <dosya>.tmp'ye savez + os.replace. Mevcut TUM anahtarlar korunur
  (yazim sonrasi ilk dosyada anahtar-kume dogrulamasi yapilir).
  channels_version eslesen dosyalar atlanir (--force ile yeniden yazilir).

Ornek:
  # kuru kosu (8 dosya):
  python extract_channels.py --data .../processed_data/validation \
      --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth --limit 8
  # gercek yazim:
  python extract_channels.py --data ... --pretrained_path ... --apply
"""
import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from GameFormer.channels import (compute_channels, compute_map_channels,
                                 NUM_CHANNELS, NUM_EVIDENCE, NUM_MAP_CHANNELS,
                                 CHANNEL_NAMES, MAP_CHANNEL_NAMES)

VERSION = "channels-v1.0"


@torch.no_grad()
def gf_top1(gameformer, batch, device, num_neighbors):
    from train_planner import extract_neighbor_top1_futures
    inputs = {k: v.to(device) for k, v in batch.items()}
    top1, _, _ = extract_neighbor_top1_futures(gameformer, gameformer.encoder(inputs), num_neighbors)
    return top1.detach().cpu()


def main(args):
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    assert files, f"npz bulunamadi: {args.data}"
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} dosya (apply={args.apply}, force={args.force}, version={VERSION})")

    from GameFormer.predictor import GameFormer
    gf = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                    neighbors=args.num_neighbors)
    gf.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    gf.to(args.device).eval()

    n_done = n_skip = n_err = 0
    fire_gt = np.zeros(NUM_CHANNELS, dtype=np.int64)
    fire_gf = np.zeros(NUM_CHANNELS, dtype=np.int64)
    mfire = np.zeros(NUM_MAP_CHANNELS, dtype=np.int64)

    pbar = tqdm(total=len(files), unit="npz", desc="extract")
    for i in range(0, len(files), args.batch_size):
        chunk = files[i:i + args.batch_size]
        pbar.update(len(chunk))
        pbar.set_postfix(islenen=n_done, atlanan=n_skip, hata=n_err)
        datas, keep = [], []
        for f in chunk:
            try:
                d = dict(np.load(f, allow_pickle=True))
            except Exception as e:
                print(f"[ERR-READ] {f}: {e}")
                n_err += 1
                continue
            if (not args.force) and str(d.get("channels_version", "")) == VERSION:
                n_skip += 1
                continue
            datas.append(d)
            keep.append(f)
        if not datas:
            continue

        N = args.num_neighbors
        nbr = torch.stack([torch.from_numpy(d["neighbor_agents_past"][:N]).float() for d in datas])
        ego = torch.stack([torch.from_numpy(d["ego_agent_past"]).float() for d in datas])
        gt = torch.stack([torch.from_numpy(d["neighbor_agents_future"][:N, :, :2]).float() for d in datas])
        ref = torch.stack([torch.from_numpy(d["c_lat_candidates"]).float() for d in datas])
        lanes = torch.stack([torch.from_numpy(d["lanes"]).float() for d in datas])
        cwalks = torch.stack([torch.from_numpy(d["crosswalks"]).float() for d in datas])
        routes = torch.stack([torch.from_numpy(d["route_lanes"]).float() for d in datas])

        top1 = gf_top1(gf, {"ego_agent_past": ego, "neighbor_agents_past": nbr,
                            "map_lanes": lanes, "map_crosswalks": cwalks,
                            "route_lanes": routes}, args.device, N)

        act_gt, ev_gt = compute_channels(nbr, ego, gt, ref)
        act_gf, ev_gf = compute_channels(nbr, ego, top1, ref)
        mact, mev = compute_map_channels(lanes, cwalks, routes, ref)

        fire_gt += act_gt.sum((0, 1)).numpy()
        fire_gf += act_gf.sum((0, 1)).numpy()
        mfire += mact.sum((0, 1)).numpy()

        for b, (f, d) in enumerate(zip(keep, datas)):
            d["channel_active_gt"] = act_gt[b].numpy()
            d["channel_evidence_gt"] = ev_gt[b].numpy().astype(np.float32)
            d["channel_active_gf"] = act_gf[b].numpy()
            d["channel_evidence_gf"] = ev_gf[b].numpy().astype(np.float32)
            d["map_channel_active"] = mact[b].numpy()
            d["map_channel_evidence"] = mev[b].numpy().astype(np.float32)
            d["channels_version"] = VERSION
            if args.apply:
                tmp = f + ".tmp.npz"
                try:
                    np.savez(tmp, **d)
                    os.replace(tmp, f)
                    if n_done == 0:  # ilk yazimda anahtar-koruma dogrulamasi
                        back = np.load(f, allow_pickle=True)
                        expected = set(d.keys())
                        got = set(back.files)
                        assert expected == got, f"ANAHTAR KAYBI: {expected ^ got}"
                        print(f"[VERIFY] ilk dosya anahtar-kume dogrulandi ({len(got)} anahtar)")
                except Exception as e:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    print(f"[ERR-WRITE] {f}: {e}")
                    n_err += 1
                    continue
            n_done += 1
    pbar.set_postfix(islenen=n_done, atlanan=n_skip, hata=n_err)
    pbar.close()

    print(f"\nBITTI: islenen={n_done}  atlanan(ayni surum)={n_skip}  hata={n_err}"
          f"  {'(KURU KOSU -- yazilmadi)' if not args.apply else ''}")
    tot = max(n_done * args.num_neighbors, 1)
    print("\nego->agent fire ozetleri (GT | GF):")
    for k in range(NUM_CHANNELS):
        print(f"  {CHANNEL_NAMES[k]:36s} {fire_gt[k]:7d} | {fire_gf[k]:7d}")
    print("ego->map fire ozeti:")
    for k in range(NUM_MAP_CHANNELS):
        print(f"  {MAP_CHANNEL_NAMES[k]:36s} {mfire[k]:7d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="islenmis npz dizini")
    p.add_argument("--pretrained_path", required=True, help="frozen GF checkpoint (GF-varyant kanallar icin)")
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="ilk N dosya (test icin)")
    p.add_argument("--apply", action="store_true", help="VERILMEZSE kuru kosu: hicbir sey yazilmaz")
    p.add_argument("--force", action="store_true", help="ayni surumu de yeniden yaz")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
