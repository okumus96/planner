import argparse
import os
import glob
import pandas as pd

# NuPlan Bileşenleri
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

def print_parquet_summary(nuboard_file_path):
    """
    .nuboard dosyasının bulunduğu dizindeki parquet dosyalarını bulur ve özetler.
    """
    base_dir = os.path.dirname(nuboard_file_path)
    
    # Tüm alt klasörlerdeki parquet dosyalarını bul
    parquet_files = glob.glob(os.path.join(base_dir, '**', '*.parquet'), recursive=True)
    
    if not parquet_files:
        print("\n[UYARI] Parquet dosyası bulunamadı. Özet gösterilemiyor.\n")
        return

    print("\n" + "="*70)
    print("📊 SİMÜLASYON METRİK ÖZETİ (Parquet Dosyalarından)")
    print("="*70)
    
    for p_file in parquet_files:
        # Sadece summary veya aggregator klasörlerindeki önemli dosyaları okuyalım
        if 'summary' in p_file or 'aggregator' in p_file or 'closed_loop' in p_file:
            try:
                df = pd.read_parquet(p_file)
                file_name = os.path.basename(p_file)
                print(f"\n--- Dosya: {file_name} ---")
                
                # NuPlan metrik yapısına göre tabloyu özetleyelim
                if 'scenario_type' in df.columns and 'metric_name' in df.columns and 'score' in df.columns:
                    # Senaryo tipi ve metrik ismine göre ortalama skorları al
                    summary_df = df.groupby(['scenario_type', 'metric_name'])['score'].mean().reset_index()
                    
                    # Okunabilirliği artırmak için tabloyu pivotla (Satırlar senaryo, Sütunlar metrikler)
                    pivot_df = summary_df.pivot(index='scenario_type', columns='metric_name', values='score')
                    
                    # Sayıları virgülden sonra 2 basamakla sınırla
                    pd.options.display.float_format = '{:.2f}'.format
                    print(pivot_df.to_string())
                else:
                    # Sütunlar farklıysa ilk 5 satırı göster
                    print(df.head())
            except Exception as e:
                print(f"[HATA] {p_file} okunamadı: {e}")
                
    print("="*70 + "\n")

def start_nuboard(file_path):
    abs_path = os.path.abspath(os.path.expanduser(file_path))
    
    if not os.path.exists(abs_path):
        print(f"Hata: Dosya bulunamadı -> {abs_path}")
        return

    # NuBoard başlatılmadan ÖNCE parquet özetini ekrana bas!
    print_parquet_summary(abs_path)

    print(f"NuBoard başlatılıyor: {abs_path}")

    # 1. Ortam değişkenlerinden yolları al (yoksa geçici '/tmp' dizinini kullan)
    data_root = os.getenv("NUPLAN_DATA_ROOT", "/tmp")
    map_root = os.getenv("NUPLAN_MAPS_ROOT", "/tmp")

    # 2. Hydra KULLANMADAN senaryo kurucuyu doğrudan biz oluşturuyoruz.
    scenario_builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root="/tmp",               
        map_version="nuplan-maps-v1.0",   
        db_files=None
    )

    # 3. Araç parametrelerini al
    vehicle_parameters = get_pacifica_parameters()

    # 4. NuBoard'u başlat
    nuboard = NuBoard(
        nuboard_paths=[abs_path],
        scenario_builder=scenario_builder,
        vehicle_parameters=vehicle_parameters,
        port_number=5009
    )
    
    print("\n--- NuBoard Sunucusu Hazır ---")
    print("Tarayıcıdan açın: http://localhost:500")
    nuboard.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NuPlan NuBoard Başlatıcı")
    parser.add_argument("--path", type=str, required=True, help="nuboard dosya yolu")
    
    args = parser.parse_args()
    start_nuboard(args.path)