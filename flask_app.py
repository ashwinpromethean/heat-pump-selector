from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import os
import io

app = Flask(__name__)
app.secret_key = os.getenv("COOKIE_KEY", "long-random-secret-key-for-production-change-this")

# ---- Auth configuration ----
def get_auth_config():
    """Get authentication config from environment variables"""
    return {
        "admin": {
            "email": os.getenv("ADMIN_EMAIL", "admin@prometheanenergy.com"),
            "password": os.getenv("ADMIN_PASSWORD", "Promethean@123"),
            "name": "Admin User"
        },
        "viewer": {
            "email": os.getenv("VIEWER_EMAIL", "common@prometheanenergy.com"),
            "password": os.getenv("VIEWER_PASSWORD", "viewer_password"),
            "name": "Viewer User"
        }
    }

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---- Data loading ----
REQUIRED_COLS = [
    "Main Model","Variant","Refrigerant","Capacity Series",
    "TC (Deg. C)","TO (Deg. C)","Qo(kW)","P(kW)","I(A)","COP",
    "mLP(kg/h)","mHP(kg/h)","tcu(deg. C)","Qsc(kW)","pm(bar)","Qac(kW)"
]

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

# Load data on startup
data_path = os.getenv("DATA_PATH", "data/models.xlsx")
try:
    df = load_excel(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")
except Exception as e:
    print(f"Failed to load data: {e}")
    df = pd.DataFrame()

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

# ---- Routes ----
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        auth_config = get_auth_config()
        for username, user_data in auth_config.items():
            if user_data['email'] == email and user_data['password'] == password:
                session['user'] = username
                session['name'] = user_data['name']
                return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    if df.empty:
        return render_template('index.html', error="Data file could not be loaded")
    
    refrigerants = sorted(df["Refrigerant"].dropna().unique().tolist())
    return render_template('index.html', 
                         refrigerants=refrigerants,
                         user_name=session.get('name', 'User'),
                         data_count=len(df))

@app.route('/api/search', methods=['POST'])
@login_required
def search_heat_pumps():
    data = request.json
    
    try:
        qo = float(data['qo'])
        refrigerant = data['refrigerant']
        heat_pump_type = data['heat_pump_type']
        k = int(data.get('k', 5))
        exact_tc = data.get('exact_tc', False)
        exact_to = data.get('exact_to', False)
        
        # Calculate TC and TO based on heat pump type
        if heat_pump_type == "Water Source":
            hot_water_temp = float(data['hot_water_temp'])
            cold_water_temp = float(data['cold_water_temp'])
            tc = hot_water_temp + 5.0
            to = cold_water_temp - 5.0
        else:  # Air Source
            ambient_temp = float(data['ambient_temp'])
            hot_water_temp = float(data['hot_water_temp'])
            tc = hot_water_temp + 5.0
            to = ambient_temp - 12.0
        
        weights = {
            "Qo(kW)": float(data.get('w_qo', 1.0)),
            "TC (Deg. C)": float(data.get('w_tc', 1.0)),
            "TO (Deg. C)": float(data.get('w_to', 1.0))
        }
        
        results = nearest_rows(
            df, qo=qo, tc=tc, to=to, refrigerant=refrigerant, k=k,
            weights=weights, exact_tc=exact_tc, exact_to=exact_to
        )
        
        if results.empty:
            return jsonify({'success': False, 'message': 'No heat pumps found—try adjusting your criteria or relax exact matches.'})
        
        # Convert DataFrame to dict for JSON response
        results_dict = results.to_dict('records')
        
        return jsonify({
            'success': True,
            'count': len(results),
            'tc': tc,
            'to': to,
            'results': results_dict
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/download', methods=['POST'])
@login_required
def download_results():
    data = request.json
    
    try:
        # Recreate the search to get results
        qo = float(data['qo'])
        refrigerant = data['refrigerant']
        heat_pump_type = data['heat_pump_type']
        k = int(data.get('k', 5))
        exact_tc = data.get('exact_tc', False)
        exact_to = data.get('exact_to', False)
        
        # Calculate TC and TO based on heat pump type
        if heat_pump_type == "Water Source":
            hot_water_temp = float(data['hot_water_temp'])
            cold_water_temp = float(data['cold_water_temp'])
            tc = hot_water_temp + 5.0
            to = cold_water_temp - 5.0
        else:  # Air Source
            ambient_temp = float(data['ambient_temp'])
            hot_water_temp = float(data['hot_water_temp'])
            tc = hot_water_temp + 5.0
            to = ambient_temp - 12.0
        
        weights = {
            "Qo(kW)": float(data.get('w_qo', 1.0)),
            "TC (Deg. C)": float(data.get('w_tc', 1.0)),
            "TO (Deg. C)": float(data.get('w_to', 1.0))
        }
        
        results = nearest_rows(
            df, qo=qo, tc=tc, to=to, refrigerant=refrigerant, k=k,
            weights=weights, exact_tc=exact_tc, exact_to=exact_to
        )
        
        if results.empty:
            return jsonify({'success': False, 'message': 'No results to download'})
        
        # Create CSV content
        csv_content = results.to_csv(index=False)
        
        # Create a BytesIO object to send as file
        output = io.BytesIO()
        output.write(csv_content.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='heat_pump_results.csv'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
