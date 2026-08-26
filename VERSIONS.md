# Model versiyonlari — flag/konfigurasyon kayitlari

## v3_latmoe (tag: `v3_latmoe`, branch: `channels`) — 2026-08-26

**Checkpoint:** `training_log/v3_latmoe/causal_epoch_12_minADE_0.7998.pth` (e12; 20 epoch
kosuldu, en iyi val e12). Backbone: `training_log/normal/model_epoch_19_valADE_1.6487.pth`
(frozen GameFormer).

**Mimari (v3 = gate+typed+dod_meta uzerine):** 4x5 GERCEK sozluk — LON4
{stop, slow, accel, maintain} (reverse->stop; remain_stopped stop'a katli), LAT5V
{turn_left, turn_right, to_left, to_right, none} (inlane -> to_*); CA sonrasi LAT SINIFI
basina 5 GMM expert (`--lat_moe 1`); q_enh + cross paylasilan; lon dallanmaz (embedding);
egitimde teacher forcing (routing+embedding GT'den) + scheduled sampling (`--ss_max 0.5`,
0->0.5 lineer rampa); inference'ta routing psi'den.

**Egitim komutu (birebir):**
```
python train_planner.py --name v3_latmoe --lat_moe 1 --ss_max 0.5 \
  --train_set .../processed_data/train --valid_set .../processed_data/validation \
  --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
  --train_epochs 20 --batch_size 32 --learning_rate 1e-4 --weight_decay 0.01 --seed 3407 \
  --graph_layers 1 --nbr_enrich 2 --num_neighbors 10 --modes 6 --dropout 0.1 \
  --gate softmax --gate_channels 1 --typed_kv 1 --channel_evidence 0 --gate_trust all \
  --dod_meta 1 --lon_merge 0 --ego_residual 0 --aligned_mode straight --ego_corridor refpath \
  --lambda_mask 0.5 --lambda_kld 1.0 --lambda_ci 0.5 --lambda_nbr 0.1 --lambda_recon 0.0 \
  --recon_drop 0.5 --device cuda:0
```
(v3 konfigurasyonuyla birebir ayni taban; tek fark `--lat_moe 1 --ss_max 0.5`.)

**Eval (b*-swap / 4x5 gorunum):**
```
python eval_bswap.py --pretrained_path <backbone> --causal_path <ckpt> \
  --valid_set .../validation --graph_layers 1 --nbr_enrich 2 --ego_residual 0 \
  --gate_channels 1 --typed_kv 1 --dod_meta 1 --lat_moe 1 --device cuda:1
```
(`--feas 1` = fizibilite dipnot tablosu; varsayilan kapali.)

**Deployment / CLS (VARSAYILAN konfig — cc_select ACIK):**
```
python run_nuplan_test.py --experiment_name closed_loop_reactive_agents \
  --config config/test14-random_reduced.yaml --data_path <splits> --map_path <maps> \
  --model_path <backbone> --causal_path <ckpt> --deploy refiner \
  --graph_layers 1 --nbr_enrich 2 --ego_residual 0 --gate_channels 1 --typed_kv 1 \
  --gate_trust all --dod_meta 1 --lat_moe 1 --cc_select 1 --device cuda:1
```
`--cc_select 1` = karar-tutarli mod secimi (6 mod icinden ilan edilen b*'a uyan en yuksek
skorlu; oncelik lon&lat -> lat -> lon -> argmax). CLS'e maliyeti olculmedi degil: OLCULDU,
sifir (0.8421 vs 0.8409).

**Sonuclar (bu ckpt; DUZELTILMIS hakemle — resmi sayilar, 2026-08-26):** val minADE 0.7998 ·
CLS reduced 0.8409 (cc kapali) / **0.8421 (cc acik)** — GF 0.8199, v3-soyu bandi 0.833-0.858 ·
agreement (4x5, cc) 99.6/93.4/92.3 · compliance (cc): lon stop 25.5 / **slow 72.6** (tanimsal
dislama: v0<1 m/s zorlamalari paydada degil — durana yavasla denmez; disla yalniz slow'da) /
accel 71.3 / maintain 87.7; lat turn_l 98.5 / turn_r 96.5 / to_l 32.9 / to_r 32.8 / none 48.6.
Dipnotlar: stop hiz-bantli (duruk %90 / 0.5-4 %37 / 4-8 %13 / >8 %0 — uretec siniri, 8-s
pencere de kurtarmiyor: %1.8); none duz-koridorda %62.7.
(Hakem duzeltmesi: turn/LC siniri koridor-bazli, GT'de sifir degisim — eski-hakem sayilarindan
farklar: lat agreement +0.8, to_* +1-2, turn -0.2, lon aynen. Ilan-kosullu agreement ve
v4_lcmoe ablasyonu: plan.md par.14+.)

**Soy:** v3 (`dodmeta_v3_egoline` e13, tag oncesi) -> v3_tf (yalniz TF ablasyonu) ->
v3_moe (aile-dallari ablasyonu) -> **v3_latmoe** (secilen). Ablasyon ckpt'leri:
`training_log/v3_tf/causal_epoch_16_minADE_1.0847.pth`,
`training_log/v3_moe/causal_epoch_16_minADE_1.0833.pth`.
