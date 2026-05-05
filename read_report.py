import pandas as pd
import glob
import os

# En son çalıştırılan testin aggregator_metric klasörünü bulur
log_dir = "/home/lt-hta-ai4/GameFormer-Planner/testing_log/closed_loop_reactive_agents/gameformer_planner/2026-04-20 13:33:10.995716"
run_name = os.path.basename(log_dir)
parquet_path = f"{log_dir}/aggregator_metric/*.parquet"

# Parquet dosyasını bul ve Pandas ile oku
files = glob.glob(parquet_path)
if not files:
    print("Metrik dosyası bulunamadı!")
else:
    df = pd.read_parquet(files[0])
    
    print("\n" + "="*50)
    print(f"TEST SONUÇLARI ({run_name})")
    print("="*50)

    cols = set(df.columns)

    # Farklı nuPlan parquet şemalarını destekle.
    if {'metric_statistic_name', 'metric_score'}.issubset(cols):
        summary = df.groupby('metric_statistic_name')['metric_score'].mean().reset_index()
        for _, row in summary.iterrows():
            metric_name = row['metric_statistic_name']
            score = row['metric_score'] * 100
            print(f"{str(metric_name).ljust(35)}: %{score:.2f}")
    elif {'metric_name', 'score'}.issubset(cols):
        summary = df.groupby('metric_name')['score'].mean().reset_index()
        for _, row in summary.iterrows():
            metric_name = row['metric_name']
            score = row['score'] * 100
            print(f"{str(metric_name).ljust(35)}: %{score:.2f}")
    else:
        # Fallback: Sayısal sütunları metrik gibi yazdır.
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            print("Beklenen metrik kolonları yok. Mevcut kolonlar:")
            print(df.columns.tolist())
        else:
            means = df[numeric_cols].mean()
            for metric_name, value in means.items():
                print(f"{str(metric_name).ljust(35)}: %{value * 100:.2f}")

    print("="*50)





