from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import os

st.set_page_config(page_title="Heat Pump Selector", layout="wide")

# ---- Auth configuration with environment variable support ----
def get_auth_config():
    """Get authentication config from environment variables in production, secrets locally"""
    if os.getenv('AWS_EXECUTION_ENV') or os.getenv('STREAMLIT_CLOUD'):  # Running on AWS or Streamlit Cloud
        return {
            "usernames": {
                "admin": {
                    "email": os.getenv("ADMIN_EMAIL", "admin@prometheanenergy.com"),
                    "first_name": "Admin",
                    "last_name": "User", 
                    "password": os.getenv("ADMIN_PASSWORD", "Promethean@123")
                },
                "viewer": {
                    "email": os.getenv("VIEWER_EMAIL", "common@prometheanenergy.com"),
                    "first_name": "Viewer",
                    "last_name": "User",
                    "password": os.getenv("VIEWER_PASSWORD", "viewer_password")
                }
            }
        }
    else:
        # Local development - use secrets.toml
        return {
            "usernames": {
                username: {
                    "email": user_data["email"],
                    "first_name": user_data["first_name"], 
                    "last_name": user_data["last_name"],
                    "password": user_data["password"]
                }
                for username, user_data in st.secrets["credentials"]["usernames"].items()
            }
        }

def get_cookie_config():
    """Get cookie config from environment variables in production, secrets locally"""
    if os.getenv('AWS_EXECUTION_ENV') or os.getenv('STREAMLIT_CLOUD'):
        return {
            "name": os.getenv("COOKIE_NAME", "modelpicker_auth"),
            "key": os.getenv("COOKIE_KEY", "long-random-secret-key-for-production"),
            "expiry_days": int(os.getenv("COOKIE_EXPIRY_DAYS", "7"))
        }
    else:
        return dict(st.secrets["cookie"])

# ---- Auth (new API: login() fills st.session_state) ----
creds = get_auth_config()
cookie = get_cookie_config()
authenticator = stauth.Authenticate(
    creds, cookie["name"], cookie["key"], cookie["expiry_days"]
    # auto_hash=True by default; set False only if you pre-hash
)

try:
    authenticator.login(location="main")
except Exception as e:
    st.error(e)
    st.stop()

auth_ok = st.session_state.get("authentication_status")
if not auth_ok:
    if auth_ok is False:
        st.error("Invalid credentials")
    else:
        st.info("Please enter your credentials")
    st.stop()

name = st.session_state.get("name")
username = st.session_state.get("username")
with st.sidebar:
    st.caption(f"Signed in as **{name}** ({username})")
    authenticator.logout(location="sidebar")

# ---- Data loading ----
REQUIRED_COLS = [
    "Main Model","Variant","Refrigerant","Capacity Series",
    "TC (Deg. C)","TO (Deg. C)","Qo(kW)","P(kW)","I(A)","COP",
    "mLP(kg/h)","mHP(kg/h)","tcu(deg. C)","Qsc(kW)","pm(bar)","Qac(kW)"
]

@st.cache_data(show_spinner=False)
def load_excel(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Convert refrigerants to lowercase for consistent matching
    df["Refrigerant"] = df["Refrigerant"].str.lower()
    
    return df

# Get data path from environment variable or secrets
def get_data_path():
    """Get data path from environment variables in production, secrets locally"""
    if os.getenv('AWS_EXECUTION_ENV') or os.getenv('STREAMLIT_CLOUD'):
        return os.getenv("DATA_PATH", "data/models.xlsx")
    else:
        return st.secrets.get("data_path", "data/models.xlsx")

data_path = get_data_path()
try:
    df = load_excel(data_path)
    st.success(f"Loaded {len(df):,} rows from {data_path}")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ---- Matching logic ----
def nearest_rows(df, qo, tc, to, refrigerant, k=5, weights=None, exact_tc=False, exact_to=False):
    # Convert refrigerant to lowercase for matching
    refrigerant_lower = refrigerant.lower()
    sub = df[df["Refrigerant"] == refrigerant_lower].copy()
    if exact_tc: sub = sub[sub["TC (Deg. C)"] == tc]
    if exact_to: sub = sub[sub["TO (Deg. C)"] == to]
    if sub.empty: return pd.DataFrame()

    cols = {"Qo(kW)": qo, "TC (Deg. C)": tc, "TO (Deg. C)": to}
    
    # Clean data: convert non-numeric values to NaN and then drop rows with NaN in key columns
    for c in cols.keys():
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    
    # Remove rows where any of the key columns have NaN values
    sub = sub.dropna(subset=list(cols.keys()))
    
    if sub.empty: 
        return pd.DataFrame()
    
    # Calculate standard deviations (now safe since we've cleaned the data)
    sigmas = {c: float(sub[c].std(ddof=0)) or 1.0 for c in cols}
    weights = weights or {c: 1.0 for c in cols}

    dist_sq = np.zeros(len(sub), dtype=float)
    for c, target in cols.items():
        z = (sub[c].astype(float) - float(target)) / (sigmas[c] if sigmas[c] > 1e-9 else 1.0)
        dist_sq += float(weights.get(c, 1.0)) * (z ** 2)

    sub["distance"] = np.sqrt(dist_sq)
    sub["ΔQo"] = (sub["Qo(kW)"] - qo).round(3)
    sub["ΔTC"] = (sub["TC (Deg. C)"] - tc).round(3)
    sub["ΔTO"] = (sub["TO (Deg. C)"] - to).round(3)

    out_cols = REQUIRED_COLS + ["ΔQo","ΔTC","ΔTO","distance"]
    return sub.sort_values(["distance","COP"], ascending=[True, False]).head(k)[out_cols]

# ---- UI ----
st.title("Heat Pump Selector")

left, right = st.columns([3,2], gap="large")
with left:
    refrigerants = sorted(df["Refrigerant"].dropna().unique().tolist())
    refrigerant = st.selectbox("Refrigerant", refrigerants)
    
    # Heat pump type selection
    heat_pump_type = st.selectbox("Heat Pump Type", ["Air Source", "Water Source"])
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        qo = st.number_input("Heating Capacity Qo (kW)", value=56.4, step=0.1, format="%.3f")
    
    if heat_pump_type == "Water Source":
        with c2: 
            hot_water_temp = st.number_input("Hot Water Temperature (°C)", value=45.0, step=0.5, format="%.1f")
        with c3: 
            cold_water_temp = st.number_input("Chilled Water Temperature (°C)", value=7.0, step=0.5, format="%.1f")
        
        # Calculate TC and TO based on water temperatures
        tc = hot_water_temp + 5.0
        to = cold_water_temp - 5.0
        
        st.info(f"Calculated TC: {tc:.1f}°C (Hot water + 5°C)")
        st.info(f"Calculated TO: {to:.1f}°C (Chilled water - 5°C)")
        
    else:  # Air Source
        with c2: 
            ambient_temp = st.number_input("Ambient Air Temperature (°C)", value=35.0, step=0.5, format="%.1f")
        with c3: 
            hot_water_temp = st.number_input("Hot Water Temperature (°C)", value=45.0, step=0.5, format="%.1f")
        
        # Calculate TC and TO for air source
        tc = hot_water_temp + 5.0
        to = ambient_temp - 12.0
        
        st.info(f"Calculated TC: {tc:.1f}°C (Hot water + 5°C)")
        st.info(f"Calculated TO: {to:.1f}°C (Ambient - 12°C)")
    
    k = st.number_input("Number of Results", value=5, min_value=1, max_value=50, step=1)

with right:
    st.markdown("**Matching options**")
    exact_tc = st.checkbox("Require exact TC match?", value=False)
    exact_to = st.checkbox("Require exact TO match?", value=False)
    with st.popover("Weights (advanced)"):
        w_qo = st.number_input("Weight Qo", value=1.0, min_value=0.0, step=0.1)
        w_tc = st.number_input("Weight TC", value=1.0, min_value=0.0, step=0.1)
        w_to = st.number_input("Weight TO", value=1.0, min_value=0.0, step=0.1)

if st.button("Find closest heat pumps", type="primary"):
    res = nearest_rows(
        df, qo=qo, tc=tc, to=to, refrigerant=refrigerant, k=int(k),
        weights={"Qo(kW)": w_qo, "TC (Deg. C)": w_tc, "TO (Deg. C)": w_to},
        exact_tc=exact_tc, exact_to=exact_to,
    )
    if res.empty:
        st.warning("No heat pumps found—try adjusting your criteria or relax exact matches.")
    else:
        st.success(f"Found {len(res)} matching heat pump(s)")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                           "heat_pump_results.csv", "text/csv")
else:
    st.info("Enter your requirements and click **Find closest heat pumps**.")