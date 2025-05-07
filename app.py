import streamlit as st
import pandas as pd
import unicodedata
from fuzzywuzzy import process
import io
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Comparador Fuzzy de Excel", layout="wide")

# Initialize session state
if 'proceed_with_duplicates' not in st.session_state:
    st.session_state.proceed_with_duplicates = False
if 'has_run_matching' not in st.session_state:
    st.session_state.has_run_matching = False
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = None

# Function to remove accents
def remove_accents(text):
    if isinstance(text, str):
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
    return text

# Function for fuzzy matching
def fuzzy_left_join(df1, df2, key1, key2, threshold=70):
    """
    Perform a left join of df2 onto df1 with fuzzy matching for inexact keys.
    
    Parameters:
    df1: First dataframe with primary key key1
    df2: Second dataframe with primary key key2
    key1: Column name for primary key in df1
    key2: Column name for primary key in df2
    threshold: Minimum similarity score (0-100) to consider a fuzzy match
    
    Returns:
    - result_df: DataFrame with joined data and match_type column
    - unmatched_keys: List of df2 keys that couldn't be matched
    """
    # Create a result dataframe starting with df1
    result_df = df1.copy()
    
    # Convert keys to strings for consistent comparison
    df1_keys = df1[key1].astype(str).tolist()
    df2_keys = df2[key2].astype(str).unique().tolist()
    
    # Initialize match tracking
    match_info = {}
    unmatched_keys = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Track match type for each df2 key
    for i, sku2 in enumerate(df2_keys):
        status_text.text(f"Procesando elemento {i+1}/{len(df2_keys)}")
        progress_bar.progress((i + 1) / len(df2_keys))
        
        # First try exact matching
        if sku2 in df1_keys:
            match_info[sku2] = {'matched_key': sku2, 'match_type': 'perfect'}
        else:
            # Try fuzzy matching
            best_match, score = process.extractOne(sku2, df1_keys)
            
            if score >= threshold:
                match_info[sku2] = {'matched_key': best_match, 'match_type': 'aprox'}
            else:
                # No good match found
                unmatched_keys.append(sku2)
                match_info[sku2] = {'matched_key': None, 'match_type': 'no match'}
    
    # Create a copy of df2 with match information
    df2_with_matches = df2.copy()
    df2_with_matches['matched_key'] = df2_with_matches[key2].astype(str).map(
        lambda x: match_info[x]['matched_key'] if x in match_info else None
    )
    df2_with_matches['match_type'] = df2_with_matches[key2].astype(str).map(
        lambda x: match_info[x]['match_type'] if x in match_info else 'no match'
    )
    
    # Clear status
    status_text.empty()
    progress_bar.empty()
    
    return df2_with_matches, unmatched_keys

# Function to handle the "proceed anyway" button click
def proceed_anyway():
    st.session_state.proceed_with_duplicates = True
    
# Function to process files and run matching
def process_and_match(seller_file, dismac_file, seller_sku_columna, target_proveedor, threshold):
    try:
        # Process seller file
        with st.spinner("Procesando Excel del Vendedor..."):
            df0_seller = pd.read_excel(seller_file)
            
            # Clean column names
            df0_seller.columns = df0_seller.columns.str.lower()
            df0_seller.columns = [remove_accents(col).lower().strip() for col in df0_seller.columns]
            
            # Ensure the specified column exists
            if seller_sku_columna.lower() not in df0_seller.columns:
                st.error(f"Columna '{seller_sku_columna}' no encontrada en el Excel del vendedor. Columnas disponibles: {', '.join(df0_seller.columns)}")
                return None
            
            keep_cols = [seller_sku_columna.lower()]
            
            df_seller = df0_seller[keep_cols].copy()
            df_seller.loc[:, 'sku_clean'] = df_seller[seller_sku_columna.lower()].astype(str).str.lower().str.strip()
            
            df_seller_sku_clean_to_original_lookup = dict(zip(df_seller['sku_clean'], df_seller[seller_sku_columna.lower()].astype(str).str.strip()))
            
            # Check for duplicates
            duplicated_mask_seller = df_seller['sku_clean'].duplicated()
            if duplicated_mask_seller.any():
                st.error("⚠️ SKUs duplicados encontrados en el Excel del Vendedor:")
                st.dataframe(df0_seller[duplicated_mask_seller.values][[seller_sku_columna.lower()]].drop_duplicates())
                has_seller_duplicates = True
            else:
                st.success("✅ No se encontraron SKUs duplicados en el Excel del Vendedor")
                has_seller_duplicates = False
        
        # Process dismac file
        with st.spinner("Procesando Excel de Dismac..."):
            df0_dismac = pd.read_excel(dismac_file)
            
            # Clean column names
            df0_dismac.columns = df0_dismac.columns.str.lower()
            df0_dismac.columns = [remove_accents(col).lower().strip() for col in df0_dismac.columns]
            
            proveedor = 'proveedor'
            sku_columna = 'codigo proveedor'
            
            # Ensure the required columns exist
            missing_cols = [col for col in [proveedor, sku_columna] if col not in df0_dismac.columns]
            if missing_cols:
                st.error(f"Columnas {', '.join(missing_cols)} no encontradas en el Excel de Dismac. Columnas disponibles: {', '.join(df0_dismac.columns)}")
                return None
            
            keep_cols = [proveedor, sku_columna]
            
            # Explicitly convert problematic columns to string type to avoid Arrow/Streamlit conversion issues
            for col in keep_cols:
                if col in df0_dismac.columns:
                    df0_dismac[col] = df0_dismac[col].astype(str)
            
            df_dismac = df0_dismac[keep_cols].copy()
            df_dismac.loc[:, proveedor] = df_dismac[proveedor].astype(str).str.lower().str.strip()
            df_dismac.loc[:, 'codigo_proveedor_clean'] = df_dismac[sku_columna].astype(str).str.lower().str.strip()
            
            # Filter out null values
            non_null_mask = df_dismac['codigo_proveedor_clean'].notna() & (df_dismac['codigo_proveedor_clean'] != "")
            df_valid = df_dismac[non_null_mask]
            df0_valid = df0_dismac[non_null_mask]
            
            # Check for duplicates
            duplicated_mask_dismac = df_valid['codigo_proveedor_clean'].duplicated()
            if duplicated_mask_dismac.any():
                st.error("⚠️ SKUs duplicados encontrados en el Excel de Dismac:")
                st.dataframe(df0_valid[duplicated_mask_dismac.values][keep_cols].drop_duplicates())
                has_dismac_duplicates = True
            else:
                st.success("✅ No se encontraron SKUs duplicados en el Excel de Dismac")
                has_dismac_duplicates = False
        
        # Check if we should proceed (no duplicates or user confirmed)
        has_duplicates = has_seller_duplicates or has_dismac_duplicates
        
        # If there are duplicates and user hasn't clicked the proceed button yet
        if has_duplicates and not st.session_state.proceed_with_duplicates:
            st.warning("⚠️ Se encontraron SKUs duplicados en los archivos Excel. Esto puede afectar los resultados de la comparación.")
            st.button("Continuar con duplicados de todos modos", on_click=proceed_anyway, key="proceed_button")
            return None
            
        # Perform fuzzy matching
        with st.spinner("Realizando coincidencia aproximada..."):
            # Filter dismac data for target proveedor
            df_dismac_subset = df_dismac[df_dismac[proveedor] == target_proveedor.lower()]
            
            if df_dismac_subset.shape[0] == 0:
                available_proveedores = df_dismac[proveedor].unique().tolist()
                st.error(f"No se encontraron registros para el proveedor '{target_proveedor}'. Proveedores disponibles: {', '.join(available_proveedores)}")
                return None
            
            # Perform fuzzy matching
            final_df, unmatched_keys = fuzzy_left_join(
                df1=df_dismac_subset,
                df2=df_seller,
                key1='codigo_proveedor_clean',
                key2='sku_clean',
                threshold=threshold
            )
            
            # Convert unmatched keys to original format
            unmatched_keys_original = [df_seller_sku_clean_to_original_lookup.get(x, x) for x in unmatched_keys]
            
            # Merge output with dismac data
            final_df_join = pd.merge(
                final_df,
                df_dismac_subset,
                left_on='matched_key',
                right_on='codigo_proveedor_clean',
                how='inner'
            )
            
            final_df_join = final_df_join[[seller_sku_columna.lower(), sku_columna, 'match_type']]
            final_df_join.loc[:, sku_columna] = final_df_join[sku_columna].astype(str).str.strip()
            final_df_join = final_df_join.rename(columns={sku_columna: 'match_valor'})
            
            # Merge with seller data
            final_df_join_2 = pd.merge(
                final_df_join,
                df0_seller,
                on=seller_sku_columna.lower(),
                how='right'
            )
            
            # Fill null values in match_type with 'no match'
            final_df_join_2['match_type'] = final_df_join_2['match_type'].fillna('no match')
            
            # Assert that total SKUs equals sum of match types
            assert len(final_df_join_2) == (
                len(final_df_join_2[final_df_join_2['match_type'] == 'perfect']) +
                len(final_df_join_2[final_df_join_2['match_type'] == 'aprox']) +
                len(final_df_join_2[final_df_join_2['match_type'] == 'no match'])
            ), "Total SKUs doesn't equal sum of match types"
            
            # Ensure all columns are string type to avoid PyArrow conversion issues
            for col in final_df_join_2.columns:
                if col != 'match_type':  # Keep match_type as is for filtering
                    final_df_join_2[col] = final_df_join_2[col].astype(str)
            
            # Return the results
            return {
                'final_results': final_df_join_2,
                'unmatched_keys': unmatched_keys_original
            }
            
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
        st.exception(e)
        return None

# App title
st.title("🔍 Comparador Fuzzy de Excel")
st.markdown("""
Esta aplicación compara SKUs entre dos archivos Excel utilizando coincidencia aproximada.
Sube tus archivos, configura los parámetros y obtén resultados de coincidencia.
""")

# Sidebar for parameters
st.sidebar.title("⚙️ Parámetros")
seller_sku_columna = st.sidebar.text_input("Nombre de Columna SKU del Vendedor", "", placeholder="Ingresa el nombre de la columna SKU del vendedor")
target_proveedor = st.sidebar.text_input("Proveedor Objetivo", "", placeholder="Ingresa el proveedor objetivo")
threshold = st.sidebar.slider("Umbral de Coincidencia Aproximada", 0, 100, 70, 
                            help="Puntuación mínima (0-100) para considerar una coincidencia aproximada")

# File uploads
st.header("📁 Subir Archivos")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Archivo Excel del Vendedor")
    seller_file = st.file_uploader("Subir archivo Excel del Vendedor", type=["xlsx", "xls"])
    if seller_file:
        st.success(f"Subido: {seller_file.name}")

with col2:
    st.subheader("Archivo Excel de Dismac")
    dismac_file = st.file_uploader("Subir archivo Excel de Dismac", type=["xlsx", "xls"])
    if dismac_file:
        st.success(f"Subido: {dismac_file.name}")

# Run button
run_matching = st.button("🚀 Ejecutar Comparación", 
                        disabled=(seller_file is None or dismac_file is None))

# Process files when button is clicked
if run_matching:
    st.session_state.has_run_matching = True
    st.session_state.matching_results = process_and_match(
        seller_file, dismac_file, seller_sku_columna, target_proveedor, threshold
    )

# Display results if available
if st.session_state.has_run_matching and st.session_state.matching_results is not None:
    results = st.session_state.matching_results
    final_df_join_2 = results['final_results']
    
    # Define match categories
    perfect_matches = final_df_join_2[final_df_join_2['match_type'] == 'perfect']
    fuzzy_matches = final_df_join_2[final_df_join_2['match_type'] == 'aprox']
    no_matches = final_df_join_2[final_df_join_2['match_type'] == 'no match']
    
    # Summary statistics
    st.subheader("📈 Estadísticas Resumidas")
    
    total_skus = len(final_df_join_2)
    perfect_matches_count = len(perfect_matches)
    fuzzy_matches_count = len(fuzzy_matches)
    no_matches_count = len(no_matches)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de SKUs", total_skus)
    
    with col2:
        st.metric("Coincidencias Perfectas", perfect_matches_count, 
                 f"{perfect_matches_count/total_skus*100:.1f}%" if total_skus > 0 else "0%")
    
    with col3:
        st.metric("Coincidencias Aproximadas", fuzzy_matches_count,
                 f"{fuzzy_matches_count/total_skus*100:.1f}%" if total_skus > 0 else "0%")
    
    with col4:
        st.metric("Sin Coincidencias", no_matches_count,
                 f"{no_matches_count/total_skus*100:.1f}%" if total_skus > 0 else "0%")
    
    # Display all results with tabs
    st.subheader("📊 Todos los Resultados")
    
    tabs = st.tabs(["Todos los Resultados", "Coincidencias Perfectas", "Coincidencias Aproximadas", "Sin Coincidencias"])
    
    # Function to safely display dataframes
    def safe_display_dataframe(df):
        # Create a copy to avoid modifying the original
        display_df = df.copy()
        # Ensure all columns are strings for display (except match_type)
        for col in display_df.columns:
            if col != 'match_type':
                display_df[col] = display_df[col].astype(str)
        st.dataframe(display_df, use_container_width=True)
    
    with tabs[0]:
        safe_display_dataframe(final_df_join_2)
    
    with tabs[1]:
        if not perfect_matches.empty:
            safe_display_dataframe(perfect_matches)
        else:
            st.info("No se encontraron coincidencias perfectas")
    
    with tabs[2]:
        if not fuzzy_matches.empty:
            safe_display_dataframe(fuzzy_matches)
        else:
            st.info("No se encontraron coincidencias aproximadas")
    
    with tabs[3]:
        if not no_matches.empty:
            safe_display_dataframe(no_matches)
        else:
            st.info("No se encontraron SKUs sin coincidencia")
    
    # Provide download link for all results - using a container to avoid rerun issues
    download_container = st.container()
    with download_container:
        st.subheader("📥 Descargar Resultados")
        
        # Create an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            final_df_join_2.to_excel(writer, index=False, sheet_name='Resultados')
        
        output.seek(0)
        
        st.download_button(
            label="Descargar Todos los Resultados como Excel",
            data=output,
            file_name="resultados_coincidencia_fuzzy.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_results"
        )

# App info in sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
## 📖 Cómo Usar
1. Sube los archivos Excel del Vendedor y Dismac
2. Configura el nombre de la columna SKU del vendedor y el proveedor objetivo
3. Ajusta el umbral de coincidencia aproximada si es necesario
4. Haz clic en "Ejecutar Comparación" para ver los resultados
""")