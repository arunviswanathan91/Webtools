#!/usr/bin/env python3
"""
Streamlit App for Publication-Quality Bar Graphs
Upload CSV files and generate customizable bar charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Page config
st.set_page_config(page_title="Bar Graph Generator", layout="wide", page_icon="📊")

# Title
st.title("📊 Publication-Quality Bar Graph Generator")
st.markdown("Upload your CSV file and generate customizable bar charts")

# Sidebar for parameters
st.sidebar.header("Bar Graph Parameters")

# Visual parameters
FIGURE_DPI = st.sidebar.slider("Figure DPI", 100, 600, 300, 50)
FIGURE_WIDTH = st.sidebar.slider("Figure Width (inches)", 4, 20, 10, 1)
FIGURE_HEIGHT = st.sidebar.slider("Figure Height (inches)", 4, 16, 6, 1)
FONT_SIZE_TITLE = st.sidebar.slider("Title Font Size", 10, 24, 16, 1)
FONT_SIZE_LABEL = st.sidebar.slider("Axis Label Font Size", 8, 20, 14, 1)
FONT_SIZE_TICK = st.sidebar.slider("Tick Font Size", 6, 16, 12, 1)

# Graph type
graph_type = st.sidebar.radio(
    "Graph Type",
    ["Vertical Bars", "Horizontal Bars", "Grouped Bars", "Stacked Bars"]
)

# Color options
st.sidebar.subheader("Colors & Style")
color_palette = st.sidebar.selectbox(
    "Color Palette",
    ["Set2", "Set3", "Pastel1", "Dark2", "tab10", "husl", "deep", "muted"]
)
show_error_bars = st.sidebar.checkbox("Show Error Bars", value=False)
show_values = st.sidebar.checkbox("Show Values on Bars", value=True)
show_grid = st.sidebar.checkbox("Show Grid", value=True)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Show data preview
        with st.expander("📊 Data Preview (first 10 rows)"):
            st.dataframe(df.head(10))
        
        # Configure bar graph
        st.subheader("📋 Configure Bar Graph")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_col = st.selectbox("Category Column (X-axis)", df.columns.tolist())
        
        with col2:
            value_col = st.selectbox("Value Column (Y-axis)", df.columns.tolist())
        
        # Optional: grouping column for grouped/stacked bars
        group_col = None
        if graph_type in ["Grouped Bars", "Stacked Bars"]:
            group_col = st.selectbox(
                "Group Column (for grouped/stacked bars)",
                ["None"] + df.columns.tolist()
            )
            if group_col == "None":
                group_col = None
        
        # Optional: error bar column
        error_col = None
        if show_error_bars:
            error_col = st.selectbox(
                "Error Column (standard deviation/error)",
                ["None"] + df.columns.tolist()
            )
            if error_col == "None":
                error_col = None
        
        # Optional: custom labels
        with st.expander("🏷️ Custom Labels"):
            x_label = st.text_input("X-axis Label", value=category_col)
            y_label = st.text_input("Y-axis Label", value=value_col)
            title = st.text_input("Graph Title", value="Bar Graph")
        
        # Generate button
        if st.button("🎨 Generate Bar Graph", type="primary"):
            with st.spinner("Generating bar graph..."):
                try:
                    # Aggregate data if needed
                    if group_col:
                        plot_data = df[[category_col, value_col, group_col]].copy()
                        if error_col:
                            plot_data[error_col] = df[error_col]
                    else:
                        # Aggregate by category
                        if error_col:
                            plot_data = df.groupby(category_col).agg({
                                value_col: 'mean',
                                error_col: 'mean'
                            }).reset_index()
                        else:
                            plot_data = df.groupby(category_col)[value_col].mean().reset_index()
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
                    
                    # Set style
                    sns.set_palette(color_palette)
                    
                    if graph_type == "Vertical Bars":
                        if group_col:
                            # Grouped bars
                            pivot_data = plot_data.pivot(index=category_col, columns=group_col, values=value_col)
                            pivot_data.plot(kind='bar', ax=ax, width=0.8)
                        else:
                            # Simple bars
                            bars = ax.bar(
                                plot_data[category_col],
                                plot_data[value_col],
                                color=sns.color_palette(color_palette, len(plot_data)),
                                edgecolor='black',
                                linewidth=1.5
                            )
                            
                            if error_col and error_col in plot_data.columns:
                                ax.errorbar(
                                    range(len(plot_data)),
                                    plot_data[value_col],
                                    yerr=plot_data[error_col],
                                    fmt='none',
                                    ecolor='black',
                                    capsize=5,
                                    capthick=2
                                )
                            
                            # Add value labels
                            if show_values:
                                for i, (bar, val) in enumerate(zip(bars, plot_data[value_col])):
                                    height = bar.get_height()
                                    ax.text(
                                        bar.get_x() + bar.get_width()/2.,
                                        height,
                                        f'{val:.2f}',
                                        ha='center',
                                        va='bottom',
                                        fontsize=10,
                                        fontweight='bold'
                                    )
                    
                    elif graph_type == "Horizontal Bars":
                        if group_col:
                            pivot_data = plot_data.pivot(index=category_col, columns=group_col, values=value_col)
                            pivot_data.plot(kind='barh', ax=ax, width=0.8)
                        else:
                            bars = ax.barh(
                                plot_data[category_col],
                                plot_data[value_col],
                                color=sns.color_palette(color_palette, len(plot_data)),
                                edgecolor='black',
                                linewidth=1.5
                            )
                            
                            if error_col and error_col in plot_data.columns:
                                ax.errorbar(
                                    plot_data[value_col],
                                    range(len(plot_data)),
                                    xerr=plot_data[error_col],
                                    fmt='none',
                                    ecolor='black',
                                    capsize=5,
                                    capthick=2
                                )
                            
                            if show_values:
                                for bar, val in zip(bars, plot_data[value_col]):
                                    width = bar.get_width()
                                    ax.text(
                                        width,
                                        bar.get_y() + bar.get_height()/2.,
                                        f'{val:.2f}',
                                        ha='left',
                                        va='center',
                                        fontsize=10,
                                        fontweight='bold'
                                    )
                    
                    elif graph_type == "Grouped Bars" and group_col:
                        pivot_data = plot_data.pivot(index=category_col, columns=group_col, values=value_col)
                        pivot_data.plot(kind='bar', ax=ax, width=0.8)
                        ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    elif graph_type == "Stacked Bars" and group_col:
                        pivot_data = plot_data.pivot(index=category_col, columns=group_col, values=value_col)
                        pivot_data.plot(kind='bar', stacked=True, ax=ax, width=0.8)
                        ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Styling
                    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL, fontweight='bold')
                    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL, fontweight='bold')
                    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
                    
                    # Grid
                    if show_grid:
                        ax.grid(True, alpha=0.3, linestyle='--', axis='y' if 'Horizontal' not in graph_type else 'x')
                        ax.set_axisbelow(True)
                    
                    # Tick styling
                    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
                    
                    # Rotate x-labels if vertical bars
                    if graph_type in ["Vertical Bars", "Grouped Bars", "Stacked Bars"]:
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    # Spine styling
                    for spine in ['top', 'right']:
                        ax.spines[spine].set_visible(False)
                    for spine in ['bottom', 'left']:
                        ax.spines[spine].set_linewidth(1.5)
                    
                    plt.tight_layout()
                    
                    # Display
                    st.pyplot(fig)
                    
                    # Show summary statistics
                    st.subheader("📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{df[value_col].mean():.3f}")
                    col2.metric("Median", f"{df[value_col].median():.3f}")
                    col3.metric("Std Dev", f"{df[value_col].std():.3f}")
                    col4.metric("Count", len(df))
                    
                    # Download options
                    st.subheader("💾 Download Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
                        buf_png.seek(0)
                        st.download_button(
                            label="📥 Download PNG",
                            data=buf_png,
                            file_name="bar_graph.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                        buf_svg.seek(0)
                        st.download_button(
                            label="📥 Download SVG",
                            data=buf_svg,
                            file_name="bar_graph.svg",
                            mime="image/svg+xml"
                        )
                    
                    with col3:
                        # Download processed data
                        csv_buffer = io.StringIO()
                        plot_data.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Data CSV",
                            data=csv_buffer.getvalue(),
                            file_name="plot_data.csv",
                            mime="text/csv"
                        )
                    
                    plt.close(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error generating bar graph: {str(e)}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to get started")
    
    # Show example data format
    with st.expander("📖 Example CSV Formats"):
        st.markdown("""
        **Simple Bar Graph:**
        ```
        Category,Value
        A,10.5
        B,15.2
        C,8.7
        D,12.3
        ```
        
        **With Error Bars:**
        ```
        Category,Value,Error
        A,10.5,1.2
        B,15.2,0.8
        C,8.7,1.5
        ```
        
        **Grouped Bars:**
        ```
        Category,Value,Group
        A,10.5,Group1
        A,12.3,Group2
        B,15.2,Group1
        B,13.1,Group2
        ```
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Tips")
st.sidebar.markdown("""
- Use horizontal bars for long category names
- Grouped bars for comparing groups
- Stacked bars for part-to-whole relationships
- Error bars show variability/uncertainty
- Export SVG for publications
""")
