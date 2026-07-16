# Bizim CausalPlanner ↔ Resmi Causal-Planner (kod) karşılaştırması

Kaynak: https://github.com/Yyb-XJTU/Causal-Planner (branch `master`, PlanTF üstüne kurulu).
İncelenen dosyalar: `src/models/causal/{lightning_trainer.py, planning_model.py, memory_causal.py,
modules/trajectory_decoder.py}`, `src/utils/traj_utils/get_decision.py`.
Bizim taraf: `GameFormer/causal_graph.py` + `train_planner.py` (frozen GameFormer backbone).

Amaç: neyi FARKLI, neyi EKSİK yapıyoruz — dönüşlerde M_cas'ın boş/dağınık ve genelde yumuşak
(peak ~0.30) olmasının kaynaklarını kod seviyesinde saptamak.

---

## 0) Paper METNİ ≠ KOD (yanlış bildiğimiz 2 şey)

- **L_KLD aslında düz cross-entropy.** `decision_loss = F.cross_entropy(decision_causal, tar_decision)`
  (trainer L211) — GT 5-sınıf manevra etiketine. Paper Eq 8'deki "−KLD(p_cas‖marjinal)" yanıltıcı.
  → Bizim CE(→m*) tercihimiz doğruymuş; sadece ETİKET farklı (aşağıda #3).
- **Resmi kodda da BACKDOOR yok.** `traj_causal_inference_loss = None` (trainer L252). Confounding
  yalnızca uniform'a itiliyor: `KL(uniform‖p_cfd) + 0.1·(−entropy)` (L222-229). → Bizim backdoor,
  onların YAPMADIĞI bir ek; confounding'i kanıtlanabilir temiz yapıyor (bdgap 0.02 vs 8.9). Eksik değil, artı.

---

## 0.5) EN KRİTİK FARK — PAYLAŞILAN karar head'i (2026-07-12, ölü entropy'nin sebebi)

- Onlar `decisioon_decoder`'ı causal VE confound için **AYNI** kullanıyor (planning_model.py L459-460):
  `decision_causal = decisioon_decoder(f_cas)`, `decision_confound = decisioon_decoder(f_cfd)`.
- Biz **iki AYRI** head kullanıyorduk (`psi_cas`, `psi_cfd`). Ayrı `psi_cfd`'nin tek görevi entropy-max →
  global optimumu "girdiyi yok say → uniform → gradyan 0" → **f_cfd hiç şekillenmez → entropy ÖLÜ**
  (log6'ya yapışık, epoch 3'ten beri). İşte bizim ölü-entropy + backdoor ekleme ihtiyacımızın kökü buydu.
- **Paylaşılan head hile yapamaz:** aynı ağ causal'dan doğru manevrayı tahmin etmek ZORUNDA (L_KLD) →
  sabit/girdi-bağımsız olamaz → confound'u uniform yapmanın tek yolu **f_cfd'yi gerçekten nötr yapmak**
  → gradyan confound feature/maskelerine akar → **entropy CANLI, confounding gerçekten eğitilir.**
- Ayrıca #2 (L_ADV): formül aynı (comp floors, excl/norm trivial softmax'ta); ama davranışı da bu head
  yüzünden farklı — paylaşılan head M_cfd'yi canlı tutunca L_ADV hareket eder, bizde M_cfd ölü → pinned.
- **DÜZELTİLDİ 2026-07-12:** `psi_cas`+`psi_cfd` → tek `self.psi`, ikisine de uygulanıyor (causal_graph.py).
  Backdoor artık gereksiz (kapalı, lambda_caus 0) — paylaşılan head onun işini doğru şekilde yapıyor.

## 1) EKSİK olanlar (etkiye göre sıralı)

### (B) HARİTA onların causal graph'ında; biz çıkardık — dönüş zayıflığının ASIL sebebi
- Maskeler İKİ kenar tipini gate'liyor: `other` (ajan→ajan) VE **`g2a` (harita/graph→ajan)**
  (trainer L232-249; planning_model `hgt_encoder` çıktısı `mask_dict`).
- Dönüşte causal faktör = şerit/yol geometrisi, bir ajan değil. Bizim ajan-only grafiğimizde dönüşte
  işaretlenecek causal düğüm YOK → M_cas boş/dağınık. (Plan yine iyi, çünkü head haritayı görüyor;
  ama GRAPH boş.) Bu bir bug değil, "agent-agent only" kararımızın doğrudan sonucu.
- **Çözüm:** haritayı gate'li causal/confound bağlamı olarak grafiğe al (g2a muadili).

### (A) EXCLUSIVITY loss eksik — yumuşaklığın (peak ~0.30) sebebi
- Onların 3 maske loss'u var (trainer L239-249, 283-312):
  - complementarity `MSE(M_cas + M_cfd, 1)`  → bizdeki L_ADV.
  - **exclusivity `MSE(M_cas · M_cfd, 0)`**   → BİZDE YOK.
  - normalization `MSE(Σ M_cas, 1) + MSE(Σ M_cfd, 1)` → bizde softmax'tan zaten var.
- Exclusivity, her kenarı "ya causal ya confound"a (çarpım→0) zorlar; iki dağılımı ayrık (disjoint)
  yapar → M_cas tepesi keskinleşir. Bizde 0.5/0.5'te oturmayı kıracak hiçbir kuvvet yok.
- **Çözüm (ilk adım):** L_EXCL = mean_j(M_cas_j · M_cfd_j) ekle, 0'a it.

### (C) Karar etiketi = semantik 5-sınıf manevra (+ senaryo sınıfı)
- `decision` ∈ {stationary, straight, turning_left, turning_right, U-turn} — GT eğrilik+işaret+yaw'dan
  (get_decision.py). Ayrıca `scenario_loss = CE(scenario_causal, GT)` ×0.2 (trainer L212).
- Biz ψ_cas'ı kazanan-mod m* (denetimsiz küme indeksi) ile eğitiyoruz. Onların etiketi dönüş yönünü
  DOĞRUDAN kodluyor — tam da zayıf olduğumuz eksen.
- **Çözüm:** ψ_cas etiketini semantik 5-sınıf manevraya çevir (+ opsiyonel senaryo).

### LSTEM zamansal hafıza (2-kare, cross-stream LSTM)
- `DualStreamProcessor` / `CrossCausalFeatureIn` (memory_causal.py); `_step_seq` 2 kare işliyor.
- Biz tek-kare. Ertelenmiş — şimdilik sorun değil.

---

## 2) FARKLI ama bizde SORUN OLMAYAN (kapatılmış) noktalar

- **"Others prediction" (`others_reg_loss`, trainer L193):** onlar tüm komşuların geleceğini de tahmin
  edip ajan özelliklerini hareket-farkında yapıyor. **Bizde GEREK YOK** — frozen GameFormer zaten
  herkesin top-1 geleceğini üretiyor ve biz komşu düğümlerine kaynaştırıyoruz
  (`extract_neighbor_top1_futures`). Aynı sinyali bedavaya alıyoruz.
- **Ağır decoder** (state/dense query + mode query + memory refinement + Laplace NLL). Graf'ın özü değil.
- **Uçtan-uca eğitim vs frozen backbone.** Bizim frozen tasarım kasıtlı (ucuz, izole).

---

## 3) FARKLI ama biz DAHA İYİ yaptığımız

- **Backdoor (CAL L_caus):** onlarda yok. Bizimki confounding'i kanıtlanabilir temiz yapıyor
  (bdgap 0.02 vs 8.9), minADE'yi bozmadan. Tut.

---

## 4) Eylem planı

- [ ] **(A) Exclusivity loss** — ucuz, 1 terim, keskinliği test et. **İLK BU.** Ölç: peak, RemoveNonCausal.
- [ ] **(B) Haritayı gate'li causal/confound düğüm olarak grafiğe al** — dönüşün asıl çözümü. A işe
      yararsa buna geç.
- [ ] **(C) Karar etiketini semantik 5-sınıf manevraya çevir** (+ opsiyonel senaryo).
- [x] Backdoor'u koru.

## Referans metrikler (mevcut, `neighbors` maske, 20ep, val minADE ~0.83)
- causal_bd (backdoor): RemoveNonCausal corr **+0.552**, ratio 5.8×, peak ~0.30, bdgap ~0.02.
- causal_faithful (backdoor yok): corr +0.538, ratio 5.2×, peak ~0.33, bdgap ~8.9.
- Viz: düz/lead-follow temiz; dönüşlerde 9/8 sahne hiçbir ajan >0.3 işaretlemiyor.
