import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

from utils.geo import vn2000_to_latlon
from utils.forecast import predict_for_station
from utils.hsi import compute_hsi

st.title("🌊 Dự báo môi trường nước cho Cá giò và Hàu khu vực biển Quảng Ninh")

# Load data of Quảng Ninh
@st.cache_data
def load_data():
    df = pd.read_csv('data/data_quang_ninh/qn_env_clean_ready.csv')
    
    # Convert Quarter column to datetime
    if 'Quarter' in df.columns:
        df['Quarter'] = pd.to_datetime(df['Quarter'], errors='coerce')
    
    # Convert VN-2000 coordinates to WGS84 (lat, lon)
    coords = df[['X', 'Y']].drop_duplicates()
    coords['lat'] = None
    coords['lon'] = None
    
    for idx, row in coords.iterrows():
        lat, lon = vn2000_to_latlon(row['X'], row['Y'])
        coords.at[idx, 'lat'] = lat
        coords.at[idx, 'lon'] = lon
    
    # Merge the converted coordinates into the original dataframe
    df = df.merge(coords[['X', 'Y', 'lat', 'lon']], on=['X', 'Y'], how='left')
    
    return df

# Load data
df = load_data()

# Get the list of unique monitoring stations
stations = df[['Station', 'Station_Name', 'lat', 'lon']].drop_duplicates()

# Forecast parameters selection
st.header("🔮 Tham số dự báo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    species_display = st.selectbox(
        "Loài",
        options=["Cá giò", "Hàu"],
        index=0
    )
    species = "cobia" if species_display == "Cá giò" else "oyster"

with col2:
    start_year = st.number_input(
        "Năm bắt đầu",
        min_value=2026,
        max_value=2030,
        value=2026,
        step=1
    )

with col3:
    start_quarter = st.selectbox(
        "Quý bắt đầu",
        options=[1, 2, 3, 4],
        index=0
    )

with col4:
    n_quarters = st.number_input(
        "Số quý dự báo",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

st.divider()

# Display the map
st.header("📍 Bản đồ các trạm quan trắc môi trường")

st.info("💡 **Hướng dẫn:** Click vào các điểm đỏ trên bản đồ để chọn trạm và xem chỉ số HSI")

# Create Folium map
center_lat = stations['lat'].mean()
center_lon = stations['lon'].mean()

# Use satellite imagery like before
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery'
)

# Add markers for each station
for idx, row in stations.iterrows():
    # Create popup content
    popup_html = f"""
    <div style="font-family: Arial; width: 200px;">
        <h4 style="color: #2E86AB; margin: 0 0 10px 0;">{row['Station']}</h4>
        <p style="margin: 5px 0;"><b>Tên:</b> {row['Station_Name']}</p>
        <p style="margin: 5px 0;"><b>Vĩ độ:</b> {row['lat']:.6f}</p>
        <p style="margin: 5px 0;"><b>Kinh độ:</b> {row['lon']:.6f}</p>
    </div>
    """
    
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=8,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{row['Station']} - {row['Station_Name']}",
        color='#C81E1E',
        fill=True,
        fillColor='#C81E1E',
        fillOpacity=0.7,
        weight=2
    ).add_to(m)

# Initialize session state for selected station FIRST
if 'selected_station' not in st.session_state:
    st.session_state.selected_station = None

# Display map and capture clicks
map_data = st_folium(
    m,
    width=None,
    height=500,
    returned_objects=["last_object_clicked"],
    key="folium_map"
)

# Handle marker click - Update session state if clicked
if map_data and map_data.get("last_object_clicked"):
    clicked_lat = map_data["last_object_clicked"]["lat"]
    clicked_lon = map_data["last_object_clicked"]["lng"]
    
    # Find the station closest to clicked location
    stations_copy = stations.copy()
    stations_copy['distance'] = ((stations_copy['lat'] - clicked_lat)**2 + (stations_copy['lon'] - clicked_lon)**2)**0.5
    closest_station = stations_copy.loc[stations_copy['distance'].idxmin(), 'Station']
    
    # Only update and rerun if different station
    if st.session_state.selected_station != closest_station:
        st.session_state.selected_station = closest_station
        st.rerun()

# Station selection for HSI calculation (placed right after map)
st.subheader("🎯 Tính toán chỉ số HSI cho trạm")

# Sort stations by number
stations_sorted = stations.copy()
stations_sorted['sort_key'] = stations_sorted['Station'].str.extract('(\d+)').astype(int)
stations_sorted = stations_sorted.sort_values('sort_key')

col_select1, col_select2 = st.columns([3, 1])

with col_select1:
    # Get default index based on session state
    default_index = 0
    if st.session_state.selected_station and st.session_state.selected_station in stations_sorted['Station'].values:
        default_index = stations_sorted['Station'].tolist().index(st.session_state.selected_station)
    else:
        # Set default to first station if not set
        st.session_state.selected_station = stations_sorted['Station'].iloc[0]
    
    # Create a search/filter box
    search_text = st.text_input(
        "🔍 Tìm kiếm trạm (nhập mã hoặc tên):",
        placeholder="Ví dụ: NB1, Cái Lân, Bãi Cháy...",
        key="search_box"
    )
    
    # Filter stations based on search
    if search_text:
        filtered_stations = stations_sorted[
            stations_sorted['Station'].str.contains(search_text, case=False, na=False) |
            stations_sorted['Station_Name'].str.contains(search_text, case=False, na=False)
        ]
        if len(filtered_stations) > 0:
            station_options = filtered_stations['Station'].tolist()
            default_index = 0  # Reset to first filtered result
        else:
            station_options = stations_sorted['Station'].tolist()
            st.warning(f"Không tìm thấy trạm nào với từ khóa '{search_text}'")
    else:
        station_options = stations_sorted['Station'].tolist()
    
    selected_station = st.selectbox(
        "Chọn trạm:",
        options=station_options,
        format_func=lambda x: f"{x} - {stations_sorted[stations_sorted['Station']==x]['Station_Name'].values[0]}",
        index=default_index,
        key="station_selector"
    )
    
    # Update session state
    st.session_state.selected_station = selected_station

with col_select2:
    calculate_btn = st.button("🔍 Tính HSI", type="primary", use_container_width=True)

# Calculate and display HSI when button is clicked or station is selected
if selected_station and (calculate_btn or 'last_station' not in st.session_state or st.session_state.last_station != selected_station):
    st.session_state.last_station = selected_station
    
    # Get station information
    station_data = df[df['Station'] == selected_station][['X', 'Y', 'Station_Name']].iloc[0]
    x_coord = station_data['X']
    y_coord = station_data['Y']
    station_name = station_data['Station_Name']
    
    with st.spinner(f'Đang tính toán HSI cho trạm {selected_station}...'):
        try:
            # Call prediction function
            forecast_df = predict_for_station(
                species=species,
                x=x_coord,
                y=y_coord,
                start_year=start_year,
                start_quarter=start_quarter,
                n_quarters=n_quarters
            )
            
            # Calculate HSI using compute_hsi
            forecast_with_hsi = compute_hsi(forecast_df, species=species)
            
            # Format results for display
            hsi_results = []
            for idx, row in forecast_with_hsi.iterrows():
                quarter_str = f"Q{int(row['quarter'])}/{int(row['year'])}"
                hsi_results.append({
                    'Thời gian': quarter_str,
                    'HSI': round(row['HSI'], 3) if not pd.isna(row['HSI']) else 'N/A',
                    'Đánh giá': row['HSI_Level']
                })
            
            hsi_df = pd.DataFrame(hsi_results)
            
            # Display results in a nice box
            st.success(f"✅ Kết quả HSI cho trạm **{selected_station}** - {station_name}")
            
            # Show parameters used
            st.caption(f"📊 Loài: **{species_display}** | Năm: **{start_year}** | Quý bắt đầu: **Q{start_quarter}** | Số quý: **{n_quarters}**")
            
            # Create tabs for chart and table view
            tab1, tab2 = st.tabs(["📈 Biểu đồ", "📋 Bảng dữ liệu"])
            
            with tab1:
                # Prepare data for chart
                chart_data = hsi_df.copy()
                chart_data['HSI_numeric'] = pd.to_numeric(chart_data['HSI'], errors='coerce')
                
                # Create line chart with Plotly
                fig = go.Figure()
                
                # Add HSI line
                fig.add_trace(go.Scatter(
                    x=chart_data['Thời gian'],
                    y=chart_data['HSI_numeric'],
                    mode='lines+markers',
                    name='HSI',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate='<b>%{x}</b><br>HSI: %{y:.3f}<br><extra></extra>'
                ))
                
                # Add threshold lines
                fig.add_hline(y=0.85, line_dash="dash", line_color="green", 
                             annotation_text="Rất phù hợp (≥0.85)", 
                             annotation_position="right")
                fig.add_hline(y=0.75, line_dash="dash", line_color="orange", 
                             annotation_text="Phù hợp (≥0.75)", 
                             annotation_position="right")
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Ít phù hợp (≥0.5)", 
                             annotation_position="right")
                
                # Customize layout
                fig.update_layout(
                    title=f"Xu hướng HSI qua các quý - {species_display}",
                    xaxis_title="Thời gian",
                    yaxis_title="Chỉ số HSI",
                    yaxis_range=[0, 1],
                    height=500,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                # Update axes
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    avg_hsi = chart_data['HSI_numeric'].mean()
                    st.metric("HSI trung bình", f"{avg_hsi:.3f}")
                
                with col_stat2:
                    min_hsi = chart_data['HSI_numeric'].min()
                    st.metric("HSI thấp nhất", f"{min_hsi:.3f}")
                
                with col_stat3:
                    max_hsi = chart_data['HSI_numeric'].max()
                    st.metric("HSI cao nhất", f"{max_hsi:.3f}")
            
            with tab2:
                # Display HSI table with color coding
                st.dataframe(
                    hsi_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "HSI": st.column_config.NumberColumn(
                            "HSI",
                            help="Chỉ số môi trường thích hợp (0-1)",
                            format="%.3f"
                        ),
                        "Đánh giá": st.column_config.TextColumn(
                            "Đánh giá",
                            help="Mức độ phù hợp"
                        )
                    }
                )
            
        except Exception as e:
            st.error(f"❌ Lỗi khi tính toán: {str(e)}")
            with st.expander("Chi tiết lỗi"):
                st.exception(e)

st.divider()

# Display the statistical information
st.subheader("📊 Thông tin dữ liệu")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Số trạm quan trắc", len(stations))

with col2:
    st.metric("Tổng số mẫu", len(df))

with col3:
    if 'Quarter' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Quarter']):
        num_years = df['Quarter'].dt.year.nunique()
    else:
        num_years = 'N/A'
    st.metric("Số năm dữ liệu", num_years)

# Display the list of monitoring stations
with st.expander("📋 Xem danh sách các trạm quan trắc"):
    # Sort by station number
    display_stations = stations.copy()
    display_stations['sort_key'] = display_stations['Station'].str.extract('(\d+)').astype(int)
    display_stations = display_stations.sort_values('sort_key')
    
    # Select and rename columns to display
    display_stations = display_stations[['Station', 'Station_Name', 'lat', 'lon']]
    display_stations.columns = ['Mã trạm', 'Tên trạm', 'Vĩ độ', 'Kinh độ']
    
    st.dataframe(
        display_stations,
        use_container_width=True,
        hide_index=True
    )