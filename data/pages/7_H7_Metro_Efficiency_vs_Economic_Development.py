#!/usr/bin/env python
# coding: utf-8

"""
Metro Efficiency vs Economic Development Analysis
===============================================
Testing the hypothesis: Metro system efficiency correlates with economic development levels,
with higher-income countries demonstrating superior metro performance due to greater investment capacity.

Author: Tan Yan Wen
Date: 22 Oct 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Metro Efficiency Analysis",
    page_icon="🚇",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 1rem;
}
.hypothesis-box {
    background-color: #f0f8ff;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #2E86AB;
    margin: 1rem 0;
}
.finding-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# World Bank income thresholds (2022, USD)
INCOME_THRESHOLDS = {
    'Low Income': 1085,
    'Lower Middle Income': 4255,
    'Upper Middle Income': 13205
    # Above 13205 is High Income
}

# Colors for economic tiers
TIER_COLORS = {
    'Low Income': '#E63946',
    'Lower Middle Income': '#F77F00', 
    'Upper Middle Income': '#FCBF49',
    'High Income': '#06A77D'
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_default_data():
    """Load default CSV files from data directory."""
    try:
        metro_data = pd.read_csv('data/metro_countries_cities.csv')
        gdp_data = pd.read_csv('data/gdp_per_capita_by_country.csv')
        return metro_data, gdp_data, None
    except FileNotFoundError as e:
        return None, None, f"Default files not found: {str(e)}"

def categorize_economic_tier(gdp):
    """Classify country into World Bank income category."""
    if gdp < INCOME_THRESHOLDS['Low Income']:
        return 'Low Income'
    elif gdp < INCOME_THRESHOLDS['Lower Middle Income']:
        return 'Lower Middle Income'
    elif gdp < INCOME_THRESHOLDS['Upper Middle Income']:
        return 'Upper Middle Income'
    else:
        return 'High Income'

@st.cache_data
def prepare_data(metro_data, gdp_data):
    """Prepare and merge the datasets."""
    # Calculate efficiency metric
    metro_data['efficiency'] = metro_data['annual_ridership_mill'] / metro_data['length_km']
    
    # Clean metro data
    metro_clean = metro_data.dropna(subset=['efficiency', 'annual_ridership_mill', 'length_km', 'stations'])
    metro_clean = metro_clean[metro_clean['efficiency'] != np.inf]
    
    # Prepare GDP data
    gdp_clean = gdp_data[['Country', '2022']].copy()
    gdp_clean.columns = ['country', 'gdp_per_capita']
    
    # Merge datasets
    merged = metro_clean.merge(gdp_clean, on='country', how='left')
    merged = merged.dropna(subset=['gdp_per_capita'])
    
    # Add economic tier classification
    merged['economic_tier'] = merged['gdp_per_capita'].apply(categorize_economic_tier)
    
    return merged

# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def create_plot1_gdp_vs_efficiency(data):
    """Plot 1: GDP per Capita vs Metro Efficiency"""
    fig = go.Figure()
    
    for tier in ['Low Income', 'Lower Middle Income', 'Upper Middle Income', 'High Income']:
        tier_data = data[data['economic_tier'] == tier]
        if not tier_data.empty:
            fig.add_trace(go.Scatter(
                x=tier_data['gdp_per_capita'],
                y=tier_data['efficiency'],
                mode='markers',
                name=tier,
                marker=dict(
                    color=TIER_COLORS[tier],
                    size=12,
                    opacity=0.7
                ),
                text=tier_data['city'],
                hovertemplate='<b>%{text}</b><br>GDP: $%{x:,.0f}<br>Efficiency: %{y:.1f} M/km<extra></extra>'
            ))
    
    # Add trend line
    correlation, p_value = stats.pearsonr(data['gdp_per_capita'], data['efficiency'])
    z = np.polyfit(data['gdp_per_capita'], data['efficiency'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=data['gdp_per_capita'].sort_values(),
        y=p(data['gdp_per_capita'].sort_values()),
        mode='lines',
        name='Trend Line',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'GDP per Capita vs Metro Efficiency<br><sub>Correlation: r = {correlation:.3f}, p = {p_value:.3f}</sub>',
        xaxis_title='GDP per Capita (USD)',
        yaxis_title='Metro Efficiency (Million riders per km)',
        height=500,
        showlegend=True
    )
    
    return fig, correlation, p_value

def create_plot2_economic_tier_comparison(data):
    """Plot 2: Metro Efficiency by Economic Tier"""
    tier_order = ['Low Income', 'Lower Middle Income', 'Upper Middle Income', 'High Income']
    
    fig = go.Figure()
    
    for tier in tier_order:
        tier_data = data[data['economic_tier'] == tier]
        if not tier_data.empty:
            fig.add_trace(go.Box(
                y=tier_data['efficiency'],
                name=tier,
                marker_color=TIER_COLORS[tier],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
    
    fig.update_layout(
        title='Metro Efficiency Distribution by Economic Development Level',
        xaxis_title='Economic Tier',
        yaxis_title='Metro Efficiency (Million riders per km)',
        height=500
    )
    
    return fig

def create_plot3_investment_vs_performance(data):
    """Plot 3: Infrastructure Investment vs Performance"""
    # Use network length as proxy for investment
    fig = go.Figure()
    
    for tier in ['Low Income', 'Lower Middle Income', 'Upper Middle Income', 'High Income']:
        tier_data = data[data['economic_tier'] == tier]
        if not tier_data.empty:
            fig.add_trace(go.Scatter(
                x=tier_data['length_km'],
                y=tier_data['annual_ridership_mill'],
                mode='markers',
                name=tier,
                marker=dict(
                    color=TIER_COLORS[tier],
                    size=10,
                    opacity=0.7
                ),
                text=tier_data['city'],
                hovertemplate='<b>%{text}</b><br>Network: %{x:.0f} km<br>Ridership: %{y:.0f}M<extra></extra>'
            ))
    
    fig.update_layout(
        title='Infrastructure Scale vs Ridership Performance',
        xaxis_title='Network Length (km)',
        yaxis_title='Annual Ridership (Millions)',
        height=500
    )
    
    return fig

def create_plot4_ridership_density(data):
    """Plot 4: Station Efficiency Analysis"""
    # Calculate ridership per station for better insight
    data['ridership_per_station'] = data['annual_ridership_mill'] / data['stations']
    
    fig = go.Figure()
    
    for tier in ['Low Income', 'Lower Middle Income', 'Upper Middle Income', 'High Income']:
        tier_data = data[data['economic_tier'] == tier]
        if not tier_data.empty:
            fig.add_trace(go.Scatter(
                x=tier_data['stations'],
                y=tier_data['ridership_per_station'],
                mode='markers',
                name=tier,
                marker=dict(
                    color=TIER_COLORS[tier],
                    size=12,
                    opacity=0.8,
                    symbol='circle'
                ),
                text=[f"{city}<br>Network: {length:.0f} km" for city, length in zip(tier_data['city'], tier_data['length_km'])],
                hovertemplate='<b>%{text}</b><br>Stations: %{x}<br>Ridership per Station: %{y:.1f}M<br>Economic Tier: ' + tier + '<extra></extra>'
            ))
    
    fig.update_layout(
        title='Station Efficiency: Ridership per Station vs Number of Stations',
        xaxis_title='Number of Stations',
        yaxis_title='Ridership per Station (Millions)',
        height=500,
        showlegend=True
    )
    
    return fig

def create_plot5_efficiency_ranking(data):
    """Plot 5: Top Metro Systems by Efficiency"""
    # Get top 15 most efficient systems
    top_systems = data.nlargest(15, 'efficiency')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_systems['efficiency'],
        y=top_systems['city'],
        orientation='h',
        marker_color=[TIER_COLORS[tier] for tier in top_systems['economic_tier']],
        text=top_systems['economic_tier'],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Efficiency: %{x:.1f} M/km<br>Tier: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Top 15 Most Efficient Metro Systems',
        xaxis_title='Metro Efficiency (Million riders per km)',
        yaxis_title='City',
        height=600,
        yaxis=dict(categoryorder='total ascending')
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚇 Metro Efficiency vs Economic Development</h1>', unsafe_allow_html=True)
    
    # Hypothesis
    st.markdown("""
    <div class="hypothesis-box">
    <h3>🎯 Research Hypothesis</h3>
    <p><strong>Metro system efficiency correlates with economic development levels, with higher-income countries 
    demonstrating superior metro performance due to greater investment capacity.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Loading Section
    st.markdown("## 📊 Data Loading")
    
    # Option to upload new files
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Metro Data (Optional)**")
        metro_file = st.file_uploader(
            "Choose metro CSV file", 
            type=['csv'], 
            key="metro",
            help="Upload updated metro_countries_cities.csv"
        )
    
    with col2:
        st.markdown("**Upload GDP Data (Optional)**")
        gdp_file = st.file_uploader(
            "Choose GDP CSV file", 
            type=['csv'], 
            key="gdp",
            help="Upload updated gdp_per_capita_by_country.csv"
        )
    
    # Load data
    if metro_file is not None and gdp_file is not None:
        # Use uploaded files
        metro_data = pd.read_csv(metro_file)
        gdp_data = pd.read_csv(gdp_file)
        st.success("✅ Using uploaded files")
    else:
        # Try to load default files
        metro_data, gdp_data, error = load_default_data()
        if error:
            st.error(f"❌ {error}")
            st.info("Please upload both CSV files to continue.")
            st.stop()
        else:
            st.info("📁 Using default data files from data/ directory")
    
    # Prepare data
    try:
        data = prepare_data(metro_data, gdp_data)
        st.success(f"✅ Data prepared successfully: {len(data)} metro systems analyzed")
    except Exception as e:
        st.error(f"❌ Error preparing data: {str(e)}")
        st.stop()
    
    # Data overview
    with st.expander("📋 Data Overview"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Systems", len(data))
        with col2:
            st.metric("Countries", data['country'].nunique())
        with col3:
            st.metric("Avg Efficiency", f"{data['efficiency'].mean():.1f}")
        with col4:
            st.metric("Total Ridership", f"{data['annual_ridership_mill'].sum():.0f}M")
        
        st.dataframe(data[['city', 'country', 'economic_tier', 'efficiency', 'gdp_per_capita']].head(10))
    
    # Analysis Plots
    st.markdown("## 📈 Analysis Results")
    
    # Plot 1: GDP vs Efficiency
    st.markdown("### 1. GDP per Capita vs Metro Efficiency")
    fig1, correlation, p_value = create_plot1_gdp_vs_efficiency(data)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot 2: Economic Tier Comparison
    st.markdown("### 2. Metro Efficiency by Economic Development Level")
    fig2 = create_plot2_economic_tier_comparison(data)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Plot 3: Investment vs Performance
    st.markdown("### 3. Infrastructure Investment vs Performance")
    fig3 = create_plot3_investment_vs_performance(data)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Plot 4: Station Efficiency
    st.markdown("### 4. Station Efficiency Analysis")
    fig4 = create_plot4_ridership_density(data)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Plot 5: Efficiency Ranking
    st.markdown("### 5. Top Performing Metro Systems")
    fig5 = create_plot5_efficiency_ranking(data)
    st.plotly_chart(fig5, use_container_width=True)
    
    # Results Section
    st.markdown("## 📊 Results & Analysis")
    
    # Calculate key statistics
    tier_stats = data.groupby('economic_tier')['efficiency'].agg(['mean', 'count']).round(2)
    
    # 1. Findings
    st.markdown("### 🔍 1. Key Findings")
    
    findings = []
    
    # Correlation analysis
    if abs(correlation) < 0.3:
        strength = "weak"
    elif abs(correlation) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    findings.append(f"**GDP-Efficiency Correlation**: {strength.title()} correlation (r = {correlation:.3f}, p = {p_value:.3f})")
    
    # Tier comparison
    high_income_avg = tier_stats.loc['High Income', 'mean'] if 'High Income' in tier_stats.index else 0
    low_income_avg = tier_stats.loc['Low Income', 'mean'] if 'Low Income' in tier_stats.index else 0
    
    if high_income_avg > low_income_avg:
        findings.append(f"**Economic Tier Performance**: High-income countries show higher average efficiency ({high_income_avg:.1f} vs {low_income_avg:.1f} M/km)")
    else:
        findings.append(f"**Economic Tier Performance**: No clear advantage for high-income countries")
    
    # Top performers analysis
    top_5 = data.nlargest(5, 'efficiency')
    high_income_in_top5 = (top_5['economic_tier'] == 'High Income').sum()
    findings.append(f"**Top Performers**: {high_income_in_top5}/5 top efficient systems are from high-income countries")
    
    for finding in findings:
        st.markdown(f"<div class='finding-box'>{finding}</div>", unsafe_allow_html=True)
    
    # 2. Target Audience
    st.markdown("### 🎯 2. Target Audience")
    
    target_audiences = [
        "**Urban Planners & Transportation Authorities**: Insights for metro system development strategies",
        "**Government Policy Makers**: Evidence for infrastructure investment decisions", 
        "**Development Finance Institutions**: Understanding relationship between economic development and transport efficiency",
        "**Academic Researchers**: Data-driven analysis of transportation economics",
        "**International Development Organizations**: Benchmarking tools for urban development projects"
    ]
    
    for audience in target_audiences:
        st.markdown(f"• {audience}")
    
    # 3. Insights & Recommendations
    st.markdown("### 💡 3. Insights & Recommendations")
    
    # Determine if hypothesis is supported
    if correlation > 0.3 and p_value < 0.05 and high_income_avg > low_income_avg * 1.2:
        hypothesis_result = "**SUPPORTED**"
        recommendation_theme = "The data supports the hypothesis that economic development correlates with metro efficiency."
    else:
        hypothesis_result = "**NOT FULLY SUPPORTED**"
        recommendation_theme = "The relationship between economic development and metro efficiency is more complex than hypothesized."
    
    st.markdown(f"**Hypothesis Result**: {hypothesis_result}")
    st.markdown(f"*{recommendation_theme}*")
    
    st.markdown("**Recommendations by Target Audience:**")
    
    recommendations = {
        "**For Urban Planners**": [
            "Focus on operational efficiency rather than just network size",
            "Study successful systems across all economic tiers for best practices",
            "Consider demand-driven expansion strategies"
        ],
        "**For Policy Makers**": [
            "Investment in metro systems requires operational excellence planning",
            "Economic development alone doesn't guarantee metro success",
            "Consider public-private partnerships for operational efficiency"
        ],
        "**For Development Organizations**": [
            "Technical assistance may be more valuable than just financial support",
            "Share operational knowledge between countries at different development levels",
            "Focus on sustainable ridership growth strategies"
        ]
    }
    
    for audience, recs in recommendations.items():
        st.markdown(audience)
        for rec in recs:
            st.markdown(f"  • {rec}")
    
    # Summary statistics table
    st.markdown("### 📋 Summary Statistics by Economic Tier")
    st.dataframe(tier_stats, use_container_width=True)

if __name__ == "__main__":
    main()