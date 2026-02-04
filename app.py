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
             # Optional mapping: allow "None"
             opts_with_none = ["(None)"] + cols
             default_idx = 0
             # Try to match, +1 because of None at 0
             match = get_default_index(cols, opt)
             if match != 0 or opt.lower() in cols[0].lower(): # Simple check if matched
                  # A bit hacky, but if get_default_index matched something useful
                  if opt.lower() in cols[match].lower():
                      default_idx = match + 1
             
             selection = st.selectbox(f"Map '{opt}' to:", opts_with_none, index=default_idx, key=f"opt_{opt}")
             if selection != "(None)":
                 mapping[opt] = selection

        # Validate
        if st.button("Validate & Preview"):
            # Create mapped dataframe with standard names
            # We rename columns provided by user to the keys in mapping
            # Invert mapping: {UserCol: StandardCol}
            # Wait, we want a DF where columns are 'gene', 'log2FC', etc.
            
            # Check for duplicate user columns mapping to different standard columns? 
            # It's allowed (e.g. x and y same column).
            
            # Construct new df
            try:
                new_data = {}
                for std_col, user_col in mapping.items():
                    new_data[std_col] = df_raw[user_col]
                
                # Keep original data for tooltips/extras? 
                # For strict schema, we just parse what we need. However, other columns might be useful.
                # But our plot functions only take arguments as series or column names.
                # The plot functions take `df` and column names.
                
                # Strategy:
                # pass the *mapped* df to the plotter.
                # But wait, logic:
                # The plot function signature is `def plot(df, x, y, ...)`
                # We call it as `plot(mapped_df, x='x', y='y', ...)`
                # Since we renamed the columns in mapped_df to be exactly the required names.
                # So we can just pass the fixed standard names as arguments!
                
                mapped_df = pd.DataFrame(new_data)
                
                # Validate
                valid, missing_req, missing_opt = SchemaValidator.validate(
                    mapped_df, 
                    plot_def.required_columns, 
                    plot_def.optional_columns
                )
                
                if valid:
                    st.success("✅ Schema Validated!")
                    is_valid = True
                else:
                    st.error(f"❌ Validation Failed. Missing: {missing_req}")
                    
            except Exception as e:
                st.error(f"Error preparing data: {e}")

with col_settings:
    st.header("🎨 Plot View")
    
    if is_valid and mapped_df is not None:
        # Settings
        with st.expander("⚙️ Appearance Settings", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                palette_names = ["npg", "nejm", "jama", "lancet", "colorblind"]
                palette = st.selectbox("Color Palette", palette_names, index=0)
                plot_title = st.text_input("Plot Title", value=plot_def.display_name)
            
            with c2:
                # Dynamic args based on plot definition?
                # We passed kwargs, so maybe we can add specific ones if needed.
                pass
        
        # Render
        if st.button("🚀 Render Plot", type="primary"):
            with st.spinner("Generating plot..."):
                try:
                    # Prepare arguments
                    # We pass column names which are now the keys of our mapping (the standard names)
                    args = {col: col for col in plot_def.required_columns}
                    
                    # Optional: only if they exist in mapped_df
                    for col in plot_def.optional_columns:
                        if col in mapped_df.columns:
                            args[col] = col
                        else:
                            args[col] = None
                            
                    # Additional args
                    args["palette"] = palette
                    args["title"] = plot_title
                    
                    # Call function
                    fig = plot_def.plot_function(mapped_df, **args)
                    
                    # Store in session state to persist? 
                    st.session_state['current_fig'] = fig
                    
                except Exception as e:
                    st.error(f"Plot Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display (Check session state)
        if 'current_fig' in st.session_state:
            fig = st.session_state['current_fig']
            
            # Determine if matplotlib or plotly
            if hasattr(fig, 'write_html'): # Plotly
                st.plotly_chart(fig, use_container_width=True)
            else: # Matplotlib
                st.pyplot(fig)
            
            st.divider()
            st.subheader("📥 Export")
            
            # Export buttons
            ec1, ec2, ec3 = st.columns(3)
            
            if hasattr(fig, 'savefig'): # Matplotlib
                with ec1:
                    buf = save_matplotlib_figure(fig, 'png', dpi=300)
                    st.download_button("Download PNG (300 DPI)", buf, "plot.png", "image/png")
                with ec2:
                    buf = save_matplotlib_figure(fig, 'svg')
                    st.download_button("Download SVG", buf, "plot.svg", "image/svg")
                with ec3:
                    buf = save_matplotlib_figure(fig, 'pdf')
                    st.download_button("Download PDF", buf, "plot.pdf", "application/pdf")
            else: # Plotly
                # Basic export for plotly, static requires kaleido
                with ec1:
                    # st.warning("Static export for interactive plots may require additional setup.")
                    pass

    else:
        st.info("Upload data and map columns to see the plot.")
        
        # Show example/placeholder image if any?
        # st.image("placeholder.png")

