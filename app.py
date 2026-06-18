import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
from dotenv import load_dotenv
from supabase import create_client
from sklearn.ensemble import RandomForestClassifier

# ── Credenciales Supabase ────────────────────────────────────────────────────
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="VarAI Detect",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 VarAI Detect")
st.markdown("**Sistema de clasificación y priorización de variantes VUS en BRCA1**")
st.markdown("---")

# ── Carga del modelo ─────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
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

# ── Carga de referencias ─────────────────────────────────────────────────────
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

# ── Funciones auxiliares ─────────────────────────────────────────────────────
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

def asignar_prioridad(prob):
    if prob >= 0.7:
        return "🔴 Alta"
    elif prob >= 0.4:
        return "🟡 Media"
    else:
        return "🟢 Baja"

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

def guardar_en_supabase(laboratorio, df_resultado):
    try:
        supabase.table("historial_analisis").insert({
            "laboratorio": laboratorio,
            "total_variantes": len(df_resultado),
            "alta":  (df_resultado["prioridad"] == "🔴 Alta").sum(),
            "media": (df_resultado["prioridad"] == "🟡 Media").sum(),
            "baja":  (df_resultado["prioridad"] == "🟢 Baja").sum(),
            "resultados": df_resultado.to_dict(orient="records")
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error al guardar en Supabase: {e}")
        return False

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📁 Subir archivo VCF",
    "📊 Ver resultados guardados",
    "📈 Historial de análisis"
])

# ── TAB 1: Subir VCF ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Subir archivo VCF con variantes del paciente")

    laboratorio = st.text_input("Nombre del laboratorio o código de muestra", 
                                 placeholder="Ej: Lab_Genomica_USIL_001")

    archivo = st.file_uploader("Selecciona tu archivo VCF", type=["vcf", "txt"])

    if archivo is not None and laboratorio:
        st.info("Procesando variantes...")

        df_vcf = parsear_vcf(archivo)
        st.write(f"**Variantes detectadas:** {len(df_vcf)}")

        df_vcf = df_vcf[
            (df_vcf["ref"].str.len() == 1) &
            (df_vcf["alt"].str.len() == 1)
        ].reset_index(drop=True)
        st.write(f"**Variantes missense procesables:** {len(df_vcf)}")

        if len(df_vcf) == 0:
            st.error("No se encontraron variantes missense en el archivo.")
        else:
            with st.spinner("Cargando referencias CADD y REVEL..."):
                df_cadd, df_revel = cargar_referencias()

            df_vcf["pos"] = df_vcf["pos"].astype(str)
            df_vcf["chr"] = df_vcf["chr"].astype(str)

            df_con_cadd = df_vcf.merge(
                df_cadd[["chr", "pos", "ref", "alt", "cadd_phred"]],
                on=["chr", "pos", "ref", "alt"], how="left"
            )
            df_con_todo = df_con_cadd.merge(
                df_revel[["grch38_pos", "ref", "alt", "REVEL"]],
                left_on=["pos", "ref", "alt"],
                right_on=["grch38_pos", "ref", "alt"], how="left"
            ).drop(columns=["grch38_pos"])

            df_procesable = df_con_todo[
                df_con_todo["cadd_phred"].notna() &
                df_con_todo["REVEL"].notna()
            ].copy().reset_index(drop=True)

            if len(df_procesable) == 0:
                st.warning("Ninguna variante tiene scores CADD y REVEL disponibles.")
            else:
                st.write(f"**Variantes con CADD y REVEL:** {len(df_procesable)}")

                progress = st.progress(0)
                status   = st.empty()
                afs = []
                for i, fila in df_procesable.iterrows():
                    af = consultar_gnomad(
                        fila["chr"], fila["pos"], fila["ref"], fila["alt"]
                    )
                    afs.append(af)
                    progress.progress((i + 1) / len(df_procesable))
                    status.text(f"Consultando gnomAD: {i+1}/{len(df_procesable)}")
                    time.sleep(0.3)

                df_procesable["af"] = afs
                status.text("✅ Consulta gnomAD completada")

                probs = modelo.predict_proba(df_procesable[FEATURES].values)[:, 1]
                df_procesable["prob_patogenica"] = probs
                df_procesable["prioridad"] = df_procesable["prob_patogenica"].apply(
                    asignar_prioridad
                )

                df_resultado = df_procesable[
                    ["chr", "pos", "ref", "alt", "cadd_phred", 
                     "REVEL", "af", "prob_patogenica", "prioridad"]
                ].sort_values("prob_patogenica", ascending=False).reset_index(drop=True)

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 Alta",  (df_resultado["prioridad"] == "🔴 Alta").sum())
                col2.metric("🟡 Media", (df_resultado["prioridad"] == "🟡 Media").sum())
                col3.metric("🟢 Baja",  (df_resultado["prioridad"] == "🟢 Baja").sum())

                st.markdown("### Tabla de resultados")
                st.dataframe(df_resultado, use_container_width=True)

                # Guardamos en Supabase
                if guardar_en_supabase(laboratorio, df_resultado):
                    st.success("✅ Resultados guardados en Supabase correctamente")

                csv = df_resultado.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar resultados CSV",
                    data=csv,
                    file_name="vus_priorizadas_varai.csv",
                    mime="text/csv"
                )

    elif archivo is not None and not laboratorio:
        st.warning("Por favor ingresa el nombre del laboratorio antes de procesar.")

# ── TAB 2: Ver resultados guardados ──────────────────────────────────────────
with tab2:
    st.subheader("Resultados del análisis previo de VUS en BRCA1")

    if os.path.exists("vus_priorizadas_varai.csv"):
        df_guardado = pd.read_csv("vus_priorizadas_varai.csv")
        df_guardado["prioridad"] = df_guardado["prob_patogenica"].apply(asignar_prioridad)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total VUS", len(df_guardado))
        col2.metric("🔴 Alta",  (df_guardado["prioridad"] == "🔴 Alta").sum())
        col3.metric("🟡 Media", (df_guardado["prioridad"] == "🟡 Media").sum())
        col4.metric("🟢 Baja",  (df_guardado["prioridad"] == "🟢 Baja").sum())

        st.markdown("---")
        prioridad_filtro = st.selectbox(
            "Filtrar por prioridad:",
            ["Todas", "🔴 Alta", "🟡 Media", "🟢 Baja"]
        )

        df_mostrar = df_guardado if prioridad_filtro == "Todas" else \
                     df_guardado[df_guardado["prioridad"] == prioridad_filtro]

        st.markdown(f"### {len(df_mostrar)} variantes")
        st.dataframe(df_mostrar, use_container_width=True)

        csv = df_mostrar.to_csv(index=False)
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv,
            file_name="vus_filtradas.csv",
            mime="text/csv"
        )
    else:
        st.warning("No hay resultados guardados. Sube un archivo VCF en la pestaña anterior.")

# ── TAB 3: Historial Supabase ─────────────────────────────────────────────────
with tab3:
    st.subheader("Historial de análisis guardados en Supabase")

    if st.button("🔄 Cargar historial"):
        try:
            respuesta = supabase.table("historial_analisis").select("*").order(
                "fecha", desc=True
            ).execute()
            df_historial = pd.DataFrame(respuesta.data)

            if len(df_historial) == 0:
                st.info("No hay análisis guardados aún.")
            else:
                st.write(f"**Total de análisis guardados:** {len(df_historial)}")
                st.dataframe(
                    df_historial[["fecha", "laboratorio", "total_variantes", 
                                  "alta", "media", "baja"]],
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error al conectar con Supabase: {e}")