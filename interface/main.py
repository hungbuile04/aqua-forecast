import streamlit as st
import pandas as pd
import pydeck as pdk

from utils.geo import vn2000_to_latlon
from utils.forecast import predict_future_metal_field_for_station, predict_future_non_metal_field_for_station
from utils.hsi import compute_hsi

st.title("🌊 Dự báo môi trường nước và tính HSI cho Quảng Ninh")