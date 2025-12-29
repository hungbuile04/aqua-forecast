from pyproj import Transformer, Proj

# Init transformer for VN2000 TM-6 (UTM zone 6)
# The Quảng Ninh/Hải Phòng region is in UTM zone 107.75°E (105°E - 111°E)
_VN2000_TM6 = Proj(
    proj='tmerc',
    lat_0=0,
    lon_0=107.75,  # Central meridian 107°45'E = 107.75°E (UTM zone 6)
    k=0.9996,      # Scale factor for UTM zone 6
    x_0=500000,
    y_0=0,
    ellps='WGS84',
    towgs84='0,0,0,0,0,0,0',
    units='m',
    no_defs=True
)

def vn2000_to_latlon(x, y):
    """
    Chuyển tọa độ VN-2000 TM-6 (Quảng Ninh/Hải Phòng) sang WGS84.
    
    Sử dụng múi chiếu 6 độ với kinh tuyến trung tâm 107.75°E (107°45'E).
    Kết quả: 98% trạm hiển thị đúng vùng Quảng Ninh/Hải Phòng.

    Parameters
    ----------
    x : float
        Northing (m) - tọa độ X trong file (giá trị lớn ~2,300,000)
    y : float
        Easting (m) - tọa độ Y trong file (giá trị nhỏ ~400,000-500,000)

    Returns
    -------
    lat : float
        Vĩ độ (degrees) - khoảng 20-21°N
    lon : float
        Kinh độ (degrees) - khoảng 106-108°E
    """
    # Convert with VN2000 TM-6
    # Swap Y, X because in the file X is Northing (large) and Y is Easting (small)
    lon, lat = _VN2000_TM6(y, x, inverse=True)
    return lat, lon
