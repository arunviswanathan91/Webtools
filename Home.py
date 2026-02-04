import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Import registry and validation
from plot_registry import PlotRegistry
from schema_validator import SchemaValidator
from utils.io import load_data
from utils.export import save_matplotlib_figure

# Import all plotters to register them
import plotters.basic
import plotters.qc
import plotters.dimred
import plotters.de
import plotters.heatmaps
import plotters.pathways
import plotters.immune
import plotters.stats
import plotters.clinical
import plotters.venn  # <--- Added Venn

# --- Page Config ---
st.set_page_config(
    page_title="Bioinformatics Plot Registry",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    h1 { color: #0e1117; font-size: 2.2rem; }
    h2 { color: #262730; font-size: 1.5rem; border-bottom: 2px solid #f0f2f6; padding-bottom: 0.5rem; }
    .stButton>button { width: 100%; border-radius: 5px; }
    .stSuccess { padding: 0.5rem; border-radius: 5px; }
    .stError { padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Navigation ---
with st.sidebar:
    st.title("📊 Plot Config")
    
    categories = PlotRegistry.get_categories()
    if not categories:
        st.error("No categories found. Check if plotters are imported correctly.")
        st.stop()
        
    selected_category = st.selectbox("1. Category", categories)
    
    plots_in_cat = PlotRegistry.get_plots_by_category(selected_category)
    plot_map = {p.display_name: p.id for p in plots_in_cat}
    selected_plot_name = st.selectbox("2. Plot Type", list(plot_map.keys()))
    
    selected_plot_id = plot_map[selected_plot_name]
    plot_def = PlotRegistry.get_plot(selected_plot_id)
    
    st.markdown("---")
    st.info(f"**{selected_plot_name}**\n\n{plot_def.description}")

# --- Main Area ---
# Landing Page Header (Unified feel from old Home.py)
# Only show if no file is uploaded? No, let's just make it the header.
st.title("📊 Bioinformatics Plot Registry")
st.markdown("### Publication-Ready Figures for Omics & Clinical Data")
st.markdown("---")

col_upload, col_settings = st.columns([1, 1.5], gap="large")

mapped_df = None
is_valid = False

with col_upload:
    st.header("📂 Data Upload")
    
    # 1. Show Schema Requirements
    with st.expander("ℹ️ Data Requirements", expanded=True):
        st.markdown(f"**Required Columns:**")
        for col in plot_def.required_columns:
            st.markdown(f"- `{col}`")
        if plot_def.optional_columns:
            st.markdown(f"**Optional Columns:**")
            for col in plot_def.optional_columns:
                st.write(f"- `{col}`")
    
    # 2. Upload
    uploaded_file = st.file_uploader("Upload CSV/TSV/TXT", type=["csv", "tsv", "txt"])
    
    df_raw = None
    if uploaded_file:
        try:
            df_raw = load_data(uploaded_file)
            st.success(f"Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
            with st.expander("Preview Raw Data"):
                st.dataframe(df_raw.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # 3. Column Mapping
    if df_raw is not None:
        st.subheader("🔗 Column Mapping")
        st.markdown("Map your columns to the required fields.")
        
        mapping = {}
        # Auto-match helper
        def get_default_index(options, key):
            key_lower = key.lower()
            for i, opt in enumerate(options):
                if key_lower in opt.lower():
                    return i
            return 0
            
        cols = df_raw.columns.tolist()
        
        # Required
        st.markdown("##### Required")
        for req in plot_def.required_columns:
            default_idx = get_default_index(cols, req)
            mapping[req] = st.selectbox(f"Map '{req}' to:", cols, index=default_idx, key=f"req_{req}")
            
        # Optional
        st.markdown("##### Optional")
        for opt in plot_def.optional_columns:
             opts_with_none = ["(None)"] + cols
             default_idx = 0
             match = get_default_index(cols, opt)
             if match != 0 or opt.lower() in cols[0].lower(): 
                  if opt.lower() in cols[match].lower():
                      default_idx = match + 1
             
             selection = st.selectbox(f"Map '{opt}' to:", opts_with_none, index=default_idx, key=f"opt_{opt}")
             if selection != "(None)":
                 mapping[opt] = selection

        # Validate
        if st.button("Validate & Preview"):
            try:
                new_data = {}
                for std_col, user_col in mapping.items():
                    new_data[std_col] = df_raw[user_col]
                
                mapped_df = pd.DataFrame(new_data)
                
                # Validate
                valid, missing_req, missing_opt = SchemaValidator.validate(
                    mapped_df, 
                    plot_def.required_columns, 
                    plot_def.optional_columns
                )
                
                if valid:
                    st.success("✅ Schema Validated!")
                    # Store valid mapped df in session state so it persists
                    st.session_state['mapped_df'] = mapped_df
                    st.session_state['is_valid'] = True
                else:
                    st.error(f"❌ Validation Failed. Missing: {missing_req}")
                    st.session_state['is_valid'] = False
                    
            except Exception as e:
                st.error(f"Error preparing data: {e}")

# Check session state for persistence
if 'mapped_df' in st.session_state and uploaded_file: # Reset if file changes? streamlit handles file_uploader reset usually
    # Basic check if mapping is still valid for current file could be complex, 
    # but for now rely on user clicking Validate if they change things.
    # Actually, simplistic: if button clicked, we store. If not, we might use stored if valid.
    # But usually `mapped_df` variable is local. Let's create it from session if available.
    if st.session_state.get('is_valid', False):
         mapped_df = st.session_state['mapped_df']
         is_valid = True

with col_settings:
    st.header("🎨 Plot View")
    
    if is_valid and mapped_df is not None:
        # Settings
        with st.expander("⚙️ Appearance Settings", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                palette_names = ["npg", "nejm", "jama", "lancet", "colorblind"]
                palette = st.selectbox("Color Palette", palette_names, index=0)
                plot_title = st.text_input("Plot Title", value=plot_def.display_name)
            
            with c2:
                # Add transparency slider for Venn?
                if "Venn" in plot_def.display_name:
                    alpha = st.slider("Transparency", 0.1, 1.0, 0.5)
                else:
                    alpha = None

        # Render
        if st.button("🚀 Render Plot", type="primary"):
            with st.spinner("Generating plot..."):
                try:
                    args = {col: col for col in plot_def.required_columns}
                    
                    for col in plot_def.optional_columns:
                        if col in mapped_df.columns:
                            args[col] = col
                        else:
                            args[col] = None
                            
                    args["palette"] = palette
                    args["title"] = plot_title
                    if alpha:
                        args["alpha"] = alpha
                    
                    fig = plot_def.plot_function(mapped_df, **args)
                    
                    st.session_state['current_fig'] = fig
                    
                except Exception as e:
                    st.error(f"Plot Error: {e}")
                    # import traceback
                    # st.code(traceback.format_exc())

        # Display
        if 'current_fig' in st.session_state:
            fig = st.session_state['current_fig']
            
            if hasattr(fig, 'write_html'):
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)
            
            st.divider()
            st.subheader("📥 Export")
            
            ec1, ec2, ec3 = st.columns(3)
            
            if hasattr(fig, 'savefig'):
                with ec1:
                    buf = save_matplotlib_figure(fig, 'png', dpi=300)
                    st.download_button("Download PNG (300 DPI)", buf, "plot.png", "image/png")
                with ec2:
                    buf = save_matplotlib_figure(fig, 'svg')
                    st.download_button("Download SVG", buf, "plot.svg", "image/svg")
                with ec3:
                    buf = save_matplotlib_figure(fig, 'pdf')
                    st.download_button("Download PDF", buf, "plot.pdf", "application/pdf")

    else:
        st.info("Upload data and map columns to see the plot.")
        
