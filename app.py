import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="VarAI Detect",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 VarAI Detect")
st.markdown("**Sistema de clasificación y priorización de variantes VUS en BRCA1**")
st.markdown("---")

# ── Carga del modelo entrenado ───────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo y el dataset de entrenamiento.
    Usamos cache_resource para no reentrenar en cada interacción.
    """
    df = pd.read_csv("brca1_dataset_final.csv")
    FEATURES = ["cadd_phred", "REVEL", "af"]
    X = df[FEATURES].values
    y = df["etiqueta"].values

    modelo = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X, y)
    return modelo, FEATURES

modelo, FEATURES = cargar_modelo()

# ── Función gnomAD ───────────────────────────────────────────────────────────
def consultar_gnomad(chr_, pos, ref, alt):
    variante_id = f"{chr_}-{pos}-{ref}-{alt}"
    query = """
    query {
      variant(variantId: "%s", dataset: gnomad_r4) {
        genome { af }
        exome  { af }
      }
    }
    """ % variante_id
    try:
        response = requests.post(
            "https://gnomad.broadinstitute.org/api",
            json={"query": query}, timeout=30
        )
        if response.status_code != 200:
            return 0.0
        data = response.json().get("data", {}).get("variant", None)
        if data is None:
            return 0.0
        genome_af = (data.get("genome") or {}).get("af", None)
        exome_af  = (data.get("exome")  or {}).get("af", None)
        if genome_af is not None:
            return genome_af
        if exome_af is not None:
            return exome_af
        return 0.0
    except Exception:
        return 0.0

# ── Función para asignar prioridad ───────────────────────────────────────────
def asignar_prioridad(prob):
    if prob >= 0.7:
        return "🔴 Alta"
    elif prob >= 0.4:
        return "🟡 Media"
    else:
        return "🟢 Baja"

# ── Función para parsear VCF ─────────────────────────────────────────────────
def parsear_vcf(archivo):
    filas = []
    for linea in archivo:
        if isinstance(linea, bytes):
            linea = linea.decode("utf-8")
        if linea.startswith("#"):
            continue
        partes = linea.strip().split("\t")
        if len(partes) < 5:
            continue
        filas.append({
            "chr": partes[0].replace("chr", ""),
            "pos": partes[1],
            "ref": partes[3],
            "alt": partes[4]
        })
    return pd.DataFrame(filas)

# ── Carga de CADD y REVEL ────────────────────────────────────────────────────
@st.cache_data
def cargar_referencias():
    df_cadd = pd.read_csv(
        "cadd_brca1.tsv", sep="\t", header=None,
        names=["chr", "pos", "ref", "alt", "raw_score", "cadd_phred"]
    )
    df_cadd["pos"] = df_cadd["pos"].astype(str)
    df_cadd["chr"] = df_cadd["chr"].astype(str)

    chunks = []
    for chunk in pd.read_csv(
        "revel_data/revel_with_transcript_ids",
        sep=",", low_memory=False, chunksize=100000
    ):
        filtrado = chunk[chunk.iloc[:, 0].astype(str) == "17"]
        if len(filtrado) > 0:
            chunks.append(filtrado)
    df_revel = pd.concat(chunks, ignore_index=True)
    df_revel["grch38_pos"] = df_revel["grch38_pos"].astype(str)
    df_revel["chr"] = df_revel["chr"].astype(str)

    return df_cadd, df_revel

# ── Tabs de la interfaz ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Subir archivo VCF", "📊 Ver resultados guardados"])

# ── TAB 1: Subir VCF ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Subir archivo VCF con variantes del paciente")
    st.markdown(
        "El sistema procesará automáticamente las variantes, "
        "consultará CADD, REVEL y gnomAD, y generará una tabla de priorización."
    )

    archivo = st.file_uploader("Selecciona tu archivo VCF", type=["vcf", "txt"])

    if archivo is not None:
        st.info("Procesando variantes...")

        df_vcf = parsear_vcf(archivo)
        st.write(f"**Variantes detectadas en el archivo:** {len(df_vcf)}")

        # Filtramos solo missense (ref y alt de un solo nucleótido)
        df_vcf = df_vcf[
            (df_vcf["ref"].str.len() == 1) &
            (df_vcf["alt"].str.len() == 1)
        ].reset_index(drop=True)
        st.write(f"**Variantes missense (procesables):** {len(df_vcf)}")

        if len(df_vcf) == 0:
            st.error("No se encontraron variantes missense en el archivo.")
        else:
            with st.spinner("Cargando referencias CADD y REVEL..."):
                df_cadd, df_revel = cargar_referencias()

            # Cruce con CADD
            df_vcf["pos"] = df_vcf["pos"].astype(str)
            df_vcf["chr"] = df_vcf["chr"].astype(str)

            df_con_cadd = df_vcf.merge(
                df_cadd[["chr", "pos", "ref", "alt", "cadd_phred"]],
                on=["chr", "pos", "ref", "alt"], how="left"
            )

            # Cruce con REVEL
            df_con_todo = df_con_cadd.merge(
                df_revel[["grch38_pos", "ref", "alt", "REVEL"]],
                left_on=["pos", "ref", "alt"],
                right_on=["grch38_pos", "ref", "alt"], how="left"
            ).drop(columns=["grch38_pos"])

            # Solo variantes con ambos scores
            df_procesable = df_con_todo[
                df_con_todo["cadd_phred"].notna() &
                df_con_todo["REVEL"].notna()
            ].copy().reset_index(drop=True)

            if len(df_procesable) == 0:
                st.warning("Ninguna variante del archivo tiene scores CADD y REVEL disponibles.")
            else:
                st.write(f"**Variantes con CADD y REVEL:** {len(df_procesable)}")

                # Consultamos gnomAD
                progress = st.progress(0)
                status   = st.empty()
                afs = []
                for i, fila in df_procesable.iterrows():
                    af = consultar_gnomad(fila["chr"], fila["pos"], fila["ref"], fila["alt"])
                    afs.append(af)
                    progress.progress((i + 1) / len(df_procesable))
                    status.text(f"Consultando gnomAD: {i+1}/{len(df_procesable)}")
                    time.sleep(0.3)

                df_procesable["af"] = afs
                status.text("✅ Consulta gnomAD completada")

                # Predicción
                probs = modelo.predict_proba(df_procesable[FEATURES].values)[:, 1]
                df_procesable["prob_patogenica"] = probs
                df_procesable["prioridad"] = df_procesable["prob_patogenica"].apply(asignar_prioridad)

                df_resultado = df_procesable[
                    ["chr", "pos", "ref", "alt", "cadd_phred", "REVEL", "af", "prob_patogenica", "prioridad"]
                ].sort_values("prob_patogenica", ascending=False).reset_index(drop=True)

                # Mostramos resumen
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 Prioridad Alta",  (df_resultado["prioridad"] == "🔴 Alta").sum())
                col2.metric("🟡 Prioridad Media", (df_resultado["prioridad"] == "🟡 Media").sum())
                col3.metric("🟢 Prioridad Baja",  (df_resultado["prioridad"] == "🟢 Baja").sum())

                st.markdown("### Tabla de resultados")
                st.dataframe(df_resultado, use_container_width=True)

                # Descarga
                csv = df_resultado.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar resultados CSV",
                    data=csv,
                    file_name="vus_priorizadas_varai.csv",
                    mime="text/csv"
                )

# ── TAB 2: Ver resultados guardados ──────────────────────────────────────────
with tab2:
    st.subheader("Resultados del análisis previo de VUS en BRCA1")

    if os.path.exists("vus_priorizadas_varai.csv"):
        df_guardado = pd.read_csv("vus_priorizadas_varai.csv")

        # Re-asignamos emojis si no los tiene
        if "prioridad" in df_guardado.columns:
            df_guardado["prioridad"] = df_guardado["prob_patogenica"].apply(asignar_prioridad)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total VUS analizadas", len(df_guardado))
        col2.metric("🔴 Prioridad Alta",  (df_guardado["prioridad"] == "🔴 Alta").sum())
        col3.metric("🟡 Prioridad Media", (df_guardado["prioridad"] == "🟡 Media").sum())
        col4.metric("🟢 Prioridad Baja",  (df_guardado["prioridad"] == "🟢 Baja").sum())

        st.markdown("---")

        # Filtro por prioridad
        prioridad_filtro = st.selectbox(
            "Filtrar por prioridad:",
            ["Todas", "🔴 Alta", "🟡 Media", "🟢 Baja"]
        )

        if prioridad_filtro == "Todas":
            df_mostrar = df_guardado
        else:
            df_mostrar = df_guardado[df_guardado["prioridad"] == prioridad_filtro]

        st.markdown(f"### Mostrando {len(df_mostrar)} variantes")
        st.dataframe(df_mostrar, use_container_width=True)

        csv = df_mostrar.to_csv(index=False)
        st.download_button(
            label="⬇️ Descargar CSV filtrado",
            data=csv,
            file_name="vus_filtradas.csv",
            mime="text/csv"
        )
    else:
        st.warning("No hay resultados guardados. Sube un archivo VCF en la pestaña anterior.")
