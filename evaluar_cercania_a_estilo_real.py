import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar métricas por video
df = pd.read_csv("comparacion_por_similitud_con_reales.csv")


# Traducción de nombres de dominio al inglés para graficar y análisis
domain_translation = {
    "reales_antiguas": "real_old",
    "degradadas_propias": "synthetic_ours",
    "degradadas_RRTN": "synthetic_RRTN",
    "degradadas_algoritmo": "synthetic_algorithm",
    "real_calidad": "real_highquality"
}

# Aplica la traducción en el DataFrame
df["domain"] = df["domain"].replace(domain_translation)


# Incluir todas las métricas a evaluar
metricas = ["lpips_mean", "brisque_mean", "entropy_mean", "flicker_index", "niqe_mean", "niqe_std"]
orden = ["real_old", "synthetic_ours", "synthetic_RRTN", "synthetic_algorithm", "real_highquality"]


# Calcular media y desviación estándar de los reales
real_df = df[df["domain"] == "reales_antiguas"]
real_mean = real_df[metricas].mean()
real_std = real_df[metricas].std()

# Calcular distancia por grupo
grupos = ["synthetic_ours", "synthetic_RRTN", "synthetic_algorithm", "real_highquality"]
resultados = {}

for grupo in grupos:
    grupo_df = df[df["domain"] == grupo]
    grupo_mean = grupo_df[metricas].mean()
    diff = (grupo_mean - real_mean).abs()
    zscore = diff / real_std
    resultados[grupo] = {
        "mean_diff": diff,
        "z_score": zscore
    }

# Evaluar quién gana por métrica
ganador_por_metrica = {}
salida = []

print("🔍 Evaluación: Qué método genera videos más similares al estilo real\n")
for metrica in metricas:
    print(f"📊 Métrica: {metrica}")
    salida.append(f"📊 Métrica: {metrica}")
    orden = sorted(resultados.items(), key=lambda x: x[1]["mean_diff"][metrica])
    for grupo, val in orden:
        delta = val["mean_diff"][metrica]
        z = val["z_score"][metrica]
        print(f"  {grupo:<15s} → Δ = {delta:.4f}, Z = {z:.2f}")
        salida.append(f"  {grupo:<15s} → Δ = {delta:.4f}, Z = {z:.2f}")
    ganador = orden[0][0]
    print(f"🏆 Ganador: {ganador}\n")
    salida.append(f"🏆 Ganador: {ganador}\n")
    ganador_por_metrica[metrica] = ganador

# =======================
# GENERAR GRÁFICAS
# =======================
df = df[df["domain"].isin(["real_old", "synthetic_ours", "synthetic_RRTN", "synthetic_algorithm", "real_highquality"])]
sns.set(style="whitegrid")

for metrica in metricas:
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(x="domain", y=metrica, data=df, hue="domain", palette="Set2", showfliers=False, legend=False)
    sns.stripplot(x="domain", y=metrica, data=df, color="black", alpha=0.4, size=4)
    plt.title(f"Distribución de {metrica}")
    plt.xlabel("Dominio")
    plt.ylabel(metrica)
    plt.tight_layout()
    plt.savefig(f"graf_{metrica}.png", dpi=300)
    plt.close()

print("📊 Gráficas guardadas como: graf_lpips_mean.png, graf_brisque_mean.png, etc.")

# =======================
# GUARDAR RESUMEN EN ARCHIVO
# =======================
with open("resumen_ganadores_por_metrica.txt", "w") as f:
    for linea in salida:
        f.write(linea + "\n")

print("📝 Resumen guardado en: resumen_ganadores_por_metrica.txt")
# =======================
# MÁS GRÁFICAS AVANZADAS
# =======================

import numpy as np

# 1. HEATMAP DE Z-SCORES POR GRUPO Y MÉTRICA
z_scores_df = pd.DataFrame({g: resultados[g]['z_score'] for g in grupos}).T
plt.figure(figsize=(10, 4))
sns.heatmap(z_scores_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Z-score por grupo y métrica respecto a reales_antiguas")
plt.ylabel("Dominio")
plt.xlabel("Métrica")
plt.tight_layout()
plt.savefig("zscore_heatmap.png", dpi=300)
plt.close()

# 2. RADAR CHART (SPIDER PLOT) DE Z-SCORE POR GRUPO
labels = metricas
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 6))
for grupo in grupos:
    stats = resultados[grupo]['z_score'].tolist()
    stats += stats[:1]
    plt.polar(angles, stats, label=grupo, linewidth=2)
plt.xticks(angles[:-1], labels)
plt.title("Perfil de Z-score por grupo")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
plt.tight_layout()
plt.savefig("radar_zscore.png", dpi=300)
plt.close()

# 3. BARPLOT AGRUPADO DE DIFERENCIAS MEDIAS
means_df = pd.DataFrame({g: resultados[g]['mean_diff'] for g in grupos})
means_df = means_df.T[metricas]  # asegurar orden
means_df.plot(kind='bar', figsize=(10,5))
plt.ylabel('Diferencia Absoluta respecto a reales_antiguas')
plt.title('Diferencia media por métrica y grupo')
plt.tight_layout()
plt.savefig("barplot_diferencia_media.png", dpi=300)
plt.close()

# 4. VIOLINPLOT POR MÉTRICA Y DOMINIO
for metrica in metricas:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="domain", y=metrica, data=df, hue="domain", palette="Set2", inner="quartile", legend=False)

    plt.title(f"Violinplot de {metrica} por dominio")
    plt.xlabel("Dominio")
    plt.ylabel(metrica)
    plt.tight_layout()
    plt.savefig(f"violin_{metrica}.png", dpi=300)
    plt.close()

# 5. MATRIZ DE CORRELACIÓN ENTRE MÉTRICAS (USANDO TODOS LOS DATOS DISPONIBLES)
corr = df[metricas].corr()
plt.figure(figsize=(7,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación entre métricas")
plt.tight_layout()
plt.savefig("correlacion_metricas.png", dpi=300)
plt.close()

print("📈 Gráficas avanzadas guardadas (zscore_heatmap.png, radar_zscore.png, barplot_diferencia_media.png, violin_*.png, correlacion_metricas.png)")



# Promedio de cada métrica por dominio
mean_metricas = df.groupby('domain')[metricas].mean()
print("\nPromedio de cada métrica por tipo de video (dominio):\n", mean_metricas)

mean_metricas.plot(kind="bar", figsize=(12,6))
plt.title("Promedio de métricas por tipo de video")
plt.ylabel("Valor promedio")
plt.xlabel("Dominio")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("barplot_promedio_metricas_por_dominio.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
sns.heatmap(mean_metricas, annot=True, fmt=".3f", cmap="viridis")
plt.title("Heatmap de promedios de métricas por dominio")
plt.ylabel("Dominio")
plt.xlabel("Métrica")
plt.tight_layout()
plt.savefig("heatmap_promedio_metricas_por_dominio.png", dpi=300)
plt.close()


labels = metricas
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(8, 6))
for idx, dominio in enumerate(mean_metricas.index):
    stats = mean_metricas.loc[dominio].tolist()
    stats += stats[:1]
    plt.polar(angles, stats, label=dominio, linewidth=2)
plt.xticks(angles[:-1], labels)
plt.title("Radar chart: promedios de métricas por dominio")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
plt.tight_layout()
plt.savefig("radar_promedio_metricas_por_dominio.png", dpi=300)
plt.close()


# =======================
# TABLAS DETALLADAS DE DISTANCIAS POR MÉTRICA (ordenadas)
# =======================

tablas_metricas = {}

for metrica in metricas:
    tabla = []
    for grupo in grupos:
        delta = resultados[grupo]["mean_diff"][metrica]
        z = resultados[grupo]["z_score"][metrica]
        tabla.append({"grupo": grupo, "delta": delta, "z": z})
    # Ordena de menor a mayor diferencia (mejor a peor)
    tabla = sorted(tabla, key=lambda x: x["delta"])
    tablas_metricas[metrica] = tabla

    # Mostrar por consola (opcional)
    print(f"\nTabla resumen para {metrica}:")
    print(f"{'Grupo':<20} {'Δ':>10} {'Z':>8}")
    for row in tabla:
        print(f"{row['grupo']:<20} {row['delta']:>10.4f} {row['z']:>8.2f}")

    # Guardar en CSV por métrica
    pd.DataFrame(tabla).to_csv(f"tabla_distancias_{metrica}.csv", index=False)

print("\nTablas de distancias y z-score guardadas en archivos tabla_distancias_<métrica>.csv")
