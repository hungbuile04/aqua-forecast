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

@st.cache_data
def load_radius_data(species):
    """Load radius data for the specified species"""
    try:
        filename = f'data/data_quang_ninh/R_{species}.csv'
        df_radius = pd.read_csv(filename)
        return df_radius
    except FileNotFoundError:
        st.warning(f"Không tìm thấy file {filename}")
        return None

@st.cache_data
def calculate_hsi_for_all_stations(species, year, quarter, station_list):
    """Calculate HSI for all stations for a specific year and quarter - optimized version"""
    import concurrent.futures
    
    def calculate_single_station(station_row):
        try:
            forecast_df = predict_for_station(
                species=species,
                x=station_row['X'],
                y=station_row['Y'],
                start_year=year,
                start_quarter=quarter,
                n_quarters=1
            )
            
            forecast_with_hsi = compute_hsi(forecast_df, species=species)
            
            if len(forecast_with_hsi) > 0:
                return (station_row['Station'], {
                    'HSI': forecast_with_hsi.iloc[0]['HSI'],
                    'HSI_Level': forecast_with_hsi.iloc[0]['HSI_Level']
                })
        except:
            pass
        return None
    
    # Use ThreadPoolExecutor for parallel processing
    hsi_results = {}
    stations_list = station_list.to_dict('records')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(calculate_single_station, stations_list)
        
    for result in results:
        if result:
            hsi_results[result[0]] = result[1]
    
    return hsi_results

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
st.header("🗺 Bản đồ các trạm quan trắc môi trường")

# Map display settings
st.subheader("⚙️ Cài đặt hiển thị bản đồ")

col_map1, col_map2, col_map3 = st.columns(3)

with col_map1:
    map_year = st.number_input(
        "Năm hiển thị",
        min_value=2026,
        max_value=2030,
        value=start_year,
        step=1,
        key="map_year"
    )

with col_map2:
    map_quarter = st.selectbox(
        "Quý hiển thị",
        options=[1, 2, 3, 4],
        index=start_quarter - 1,
        key="map_quarter"
    )

with col_map3:
    show_hsi = st.checkbox(
        "Hiển thị HSI",
        value=True,
        help="Tính toán và hiển thị HSI cho tất cả các trạm"
    )

# Load radius data based on selected species
df_radius = load_radius_data(species)

# Calculate HSI for all stations if needed
hsi_data = {}
if show_hsi:
    with st.spinner('Đang tính toán HSI cho các trạm...'):
        stations_unique = df[['Station', 'X', 'Y']].drop_duplicates()
        hsi_data = calculate_hsi_for_all_stations(species, map_year, map_quarter, stations_unique)

st.info("💡 **Hướng dẫn:** Click vào các điểm đỏ trên bản đồ để chọn trạm và xem chi tiết. Vòng tròn màu xanh biểu thị vùng áp dụng kết quả dự báo cho Q{}/{}. Hover chuột để xem thông tin nhanh.".format(map_quarter, map_year))

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

# Add radius circles first (so they appear below markers)
if df_radius is not None:
    # Filter radius data for the selected map display period
    radius_filtered = df_radius[
        (df_radius['year'] == map_year) & 
        (df_radius['quarter'] == map_quarter)
    ].copy()
    
    # Merge with station coordinates
    radius_filtered = radius_filtered.merge(
        stations[['Station', 'lat', 'lon']], 
        left_on='station', 
        right_on='Station', 
        how='left'
    )
    
    # Add circles for each station
    for idx, row in radius_filtered.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']) and pd.notna(row['R_km']):
            # Convert km to meters for folium Circle
            radius_m = row['R_km'] * 1000
            
            # Create circle
            folium.Circle(
                location=[row['lat'], row['lon']],
                radius=radius_m,
                color='#2E86AB',
                fill=True,
                fillColor='#2E86AB',
                fillOpacity=0.15,
                weight=2,
                opacity=0.5,
                popup=folium.Popup(
                    f"<b>{row['station']}</b><br>Bán kính: {row['R_km']} km<br>Q{row['quarter']}/{row['year']}",
                    max_width=200
                ),
                tooltip=f"{row['station']}: R = {row['R_km']} km"
            ).add_to(m)

# Add markers for each station (on top of circles)
for idx, row in stations.iterrows():
    # Get radius info if available
    radius_info = ""
    if df_radius is not None:
        station_radius = df_radius[
            (df_radius['station'] == row['Station']) &
            (df_radius['year'] == map_year) &
            (df_radius['quarter'] == map_quarter)
        ]
        if len(station_radius) > 0:
            r_km = station_radius.iloc[0]['R_km']
            radius_info = f"<p style='margin: 5px 0;'><b>Bán kính áp dụng:</b> {r_km} km</p>"
    
    # Get HSI info if available
    hsi_info = ""
    hsi_tooltip = ""
    marker_color = '#C81E1E'  # Default red
    
    if row['Station'] in hsi_data:
        hsi_value = hsi_data[row['Station']]['HSI']
        hsi_level = hsi_data[row['Station']]['HSI_Level']
        
        hsi_info = f"""
        <p style='margin: 5px 0;'><b>HSI (Q{map_quarter}/{map_year}):</b> {hsi_value:.3f}</p>
        <p style='margin: 5px 0;'><b>Đánh giá:</b> {hsi_level}</p>
        """
        
        hsi_tooltip = f" | HSI: {hsi_value:.3f} ({hsi_level})"
        
        # Color code based on HSI value
        if hsi_value >= 0.85:
            marker_color = '#28a745'  # Green - Very suitable
        elif hsi_value >= 0.75:
            marker_color = '#ffc107'  # Yellow/Orange - Suitable
        elif hsi_value >= 0.5:
            marker_color = '#fd7e14'  # Orange - Less suitable
        else:
            marker_color = '#dc3545'  # Red - Not suitable
    
    # Create popup content
    popup_html = f"""
    <div style="font-family: Arial; width: 240px;">
        <h4 style="color: #2E86AB; margin: 0 0 10px 0;">{row['Station']}</h4>
        <p style="margin: 5px 0;"><b>Tên:</b> {row['Station_Name']}</p>
        <p style="margin: 5px 0;"><b>Vĩ độ:</b> {row['lat']:.6f}</p>
        <p style="margin: 5px 0;"><b>Kinh độ:</b> {row['lon']:.6f}</p>
        {radius_info}
        {hsi_info}
    </div>
    """
    
    tooltip_text = f"{row['Station']} - {row['Station_Name']}{hsi_tooltip}"
    
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=8,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=tooltip_text,
        color=marker_color,
        fill=True,
        fillColor=marker_color,
        fillOpacity=0.7,
        weight=2
    ).add_to(m)

# Add legend to map
if show_hsi:
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin: 0 0 10px 0; font-weight: bold;">Chỉ số HSI:</p>
    <p style="margin: 5px 0;"><span style="color: #28a745;">●</span> Rất phù hợp (≥0.85)</p>
    <p style="margin: 5px 0;"><span style="color: #ffc107;">●</span> Phù hợp (≥0.75)</p>
    <p style="margin: 5px 0;"><span style="color: #fd7e14;">●</span> Ít phù hợp (≥0.5)</p>
    <p style="margin: 5px 0;"><span style="color: #dc3545;">●</span> Không phù hợp (<0.5)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

# Initialize session state for selected station FIRST
if 'selected_station' not in st.session_state:
    st.session_state.selected_station = None

# Display map and capture clicks
map_data = st_folium(
    m,
    width=None,
    height=500,
    returned_objects=["last_object_clicked"],
    key=f"folium_map_{map_year}_{map_quarter}_{species}"
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

st.divider()

# Station selection for HSI calculation (placed right after map)
st.subheader("🎯 Tính toán chỉ số HSI chi tiết cho trạm")

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
    calculate_btn = st.button("📊 Tính HSI", type="primary", use_container_width=True)

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
            
            # Get radius information for each forecasted quarter
            if df_radius is not None:
                radius_info_list = []
                for idx, row in forecast_with_hsi.iterrows():
                    station_radius = df_radius[
                        (df_radius['station'] == selected_station) &
                        (df_radius['year'] == int(row['year'])) &
                        (df_radius['quarter'] == int(row['quarter']))
                    ]
                    if len(station_radius) > 0:
                        radius_info_list.append(station_radius.iloc[0]['R_km'])
                    else:
                        radius_info_list.append(None)
                forecast_with_hsi['R_km'] = radius_info_list
            
            # Format results for display
            hsi_results = []
            for idx, row in forecast_with_hsi.iterrows():
                quarter_str = f"Q{int(row['quarter'])}/{int(row['year'])}"
                result_dict = {
                    'Thời gian': quarter_str,
                    'HSI': round(row['HSI'], 3) if not pd.isna(row['HSI']) else 'N/A',
                    'Đánh giá': row['HSI_Level']
                }
                
                # Add radius info if available
                if 'R_km' in row and pd.notna(row['R_km']):
                    result_dict['Bán kính (km)'] = row['R_km']
                
                hsi_results.append(result_dict)
            
            hsi_df = pd.DataFrame(hsi_results)
            
            # Display results in a nice box
            st.success(f"✅ Kết quả HSI cho trạm **{selected_station}** - {station_name}")
            
            # Show parameters used
            st.caption(f"📊 Loài: **{species_display}** | Năm: **{start_year}** | Quý bắt đầu: **Q{start_quarter}** | Số quý: **{n_quarters}**")
            
            # Show radius note if available
            if 'Bán kính (km)' in hsi_df.columns:
                st.info("ℹ️ **Bán kính áp dụng** là khoảng cách từ trạm quan trắc mà kết quả dự báo HSI có thể áp dụng được.")
            
            # Create tabs for chart and table view
            tab1, tab2, tab3 = st.tabs(["📈 Biểu đồ HSI", "🌡️ Biểu đồ các thông số môi trường", "📋 Bảng dữ liệu"])
            
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
                st.markdown("### 🌡️ Các thông số môi trường dự báo")
                
                # Get environmental parameters from forecast_df
                # Common parameters to visualize
                param_names = {
                    'temp': 'Nhiệt độ (°C)',
                    'salinity': 'Độ mặn (‰)',
                    'DO': 'Oxy hòa tan (mg/L)',
                    'pH': 'pH',
                    'turbidity': 'Độ đục (NTU)',
                    'chlorophyll': 'Chlorophyll-a (μg/L)',
                    'NH4': 'Amoni - NH4+ (mg/L)',
                    'NO3': 'Nitrat - NO3- (mg/L)',
                    'PO4': 'Phosphat - PO43- (mg/L)'
                }
                
                # Filter only available parameters
                available_params = [col for col in forecast_with_hsi.columns if col in param_names.keys()]
                
                if len(available_params) > 0:
                    # Let user select parameters to display
                    selected_params = st.multiselect(
                        "Chọn các thông số để hiển thị:",
                        options=available_params,
                        default=available_params[:3] if len(available_params) >= 3 else available_params,
                        format_func=lambda x: param_names.get(x, x)
                    )
                    
                    if selected_params:
                        # Create time labels
                        time_labels = [f"Q{int(row['quarter'])}/{int(row['year'])}" 
                                     for _, row in forecast_with_hsi.iterrows()]
                        
                        # Create subplots
                        num_params = len(selected_params)
                        cols_per_row = 2
                        num_rows = (num_params + cols_per_row - 1) // cols_per_row
                        
                        from plotly.subplots import make_subplots
                        
                        fig_env = make_subplots(
                            rows=num_rows,
                            cols=cols_per_row,
                            subplot_titles=[param_names.get(p, p) for p in selected_params],
                            vertical_spacing=0.12,
                            horizontal_spacing=0.1
                        )
                        
                        for idx, param in enumerate(selected_params):
                            row = (idx // cols_per_row) + 1
                            col = (idx % cols_per_row) + 1
                            
                            values = forecast_with_hsi[param].values
                            
                            fig_env.add_trace(
                                go.Scatter(
                                    x=time_labels,
                                    y=values,
                                    mode='lines+markers',
                                    name=param_names.get(param, param),
                                    line=dict(width=2),
                                    marker=dict(size=8),
                                    showlegend=False,
                                    hovertemplate='<b>%{x}</b><br>Giá trị: %{y:.2f}<br><extra></extra>'
                                ),
                                row=row,
                                col=col
                            )
                            
                            # Update axes
                            fig_env.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
                            fig_env.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
                        
                        # Update layout
                        fig_env.update_layout(
                            height=300 * num_rows,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig_env, use_container_width=True)
                        
                        # Show statistics table for selected parameters
                        st.markdown("#### 📊 Thống kê các thông số")
                        stats_data = []
                        for param in selected_params:
                            values = forecast_with_hsi[param].values
                            stats_data.append({
                                'Thông số': param_names.get(param, param),
                                'Trung bình': f"{values.mean():.2f}",
                                'Min': f"{values.min():.2f}",
                                'Max': f"{values.max():.2f}",
                                'Độ lệch chuẩn': f"{values.std():.2f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Vui lòng chọn ít nhất một thông số để hiển thị.")
                else:
                    st.warning("Không tìm thấy thông số môi trường trong dữ liệu dự báo.")
            
            with tab3:
                # Display HSI table with color coding
                column_config = {
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
                
                # Add radius column config if available
                if 'Bán kính (km)' in hsi_df.columns:
                    column_config["Bán kính (km)"] = st.column_config.NumberColumn(
                        "Bán kính (km)",
                        help="Vùng áp dụng kết quả dự báo",
                        format="%.1f"
                    )
                
                st.dataframe(
                    hsi_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
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