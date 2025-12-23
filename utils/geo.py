from pyproj import Transformer

# Khởi tạo transformer 1 lần (tối ưu)
_VN2000_TO_WGS84 = Transformer.from_crs(
    "EPSG:3405",   # VN-2000 / UTM zone 48N
    "EPSG:4326",   # WGS84 (lat, lon)
    always_xy=True
)

def vn2000_to_latlon(x, y):
    """
    Chuyển tọa độ VN-2000 (Quảng Ninh) sang WGS84.

    Parameters
    ----------
    x : float
        Northing (m)
    y : float
        Easting (m)

    Returns
    -------
    lat : float
    lon : float
    """
    lon, lat = _VN2000_TO_WGS84.transform(y, x)
    return lat, lon
