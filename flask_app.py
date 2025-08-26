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
    
    # Load the Excel file
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Clean the data properly
    print("Cleaning data...")
    
    # Define numeric columns that need cleaning
    numeric_cols = ["TC (Deg. C)", "TO (Deg. C)", "Qo(kW)", "P(kW)", "I(A)", "COP", 
                   "mLP(kg/h)", "mHP(kg/h)", "tcu(deg. C)", "Qsc(kW)", "pm(bar)", "Qac(kW)"]
    
    # Replace "-" and other non-numeric values with NaN, then convert to numeric
    for col in numeric_cols:
        if col in df.columns:
            # Replace "-" and empty strings with NaN
            df[col] = df[col].replace(["-", "", " "], np.nan)
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert refrigerants to lowercase for consistent matching
    df["Refrigerant"] = df["Refrigerant"].str.lower().str.strip()
    
    # Remove rows where critical columns (Qo, TC, TO) are all NaN or zero
    critical_cols = ["Qo(kW)", "TC (Deg. C)", "TO (Deg. C)"]
    df = df.dropna(subset=critical_cols, how='all')
    
    # Remove rows where Qo is 0 or negative (not useful for matching)
    df = df[(df["Qo(kW)"] > 0) & (df["Qo(kW)"].notna())]
    
    print(f"After cleaning: {len(df)} valid rows with non-zero Qo values")
    print(f"Qo range: {df['Qo(kW)'].min():.1f} - {df['Qo(kW)'].max():.1f} kW")
    
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
    """
    Find heat pumps closest to the specified criteria, prioritizing Qo (heating capacity) matching.
    """
    # Convert refrigerant to lowercase for matching
    refrigerant_lower = refrigerant.lower().strip()
    
    # Filter by refrigerant first
    sub = df[df["Refrigerant"] == refrigerant_lower].copy()
    if sub.empty:
        print(f"No heat pumps found for refrigerant: {refrigerant}")
        return pd.DataFrame()
    
    print(f"Found {len(sub)} heat pumps with refrigerant {refrigerant}")
    
    # Apply exact temperature constraints if requested
    if exact_tc:
        sub = sub[sub["TC (Deg. C)"] == tc]
        print(f"After exact TC filter ({tc}°C): {len(sub)} heat pumps")
    if exact_to:
        sub = sub[sub["TO (Deg. C)"] == to]
        print(f"After exact TO filter ({to}°C): {len(sub)} heat pumps")
    
    if sub.empty:
        print("No heat pumps found after applying exact temperature filters")
        return pd.DataFrame()
    
    # Ensure we have valid numeric data for key columns
    key_cols = ["Qo(kW)", "TC (Deg. C)", "TO (Deg. C)"]
    sub = sub.dropna(subset=key_cols)
    
    if sub.empty:
        print("No heat pumps found with valid numeric data for key columns")
        return pd.DataFrame()
    
    # Set default weights (prioritize Qo matching)
    if weights is None:
        weights = {"Qo(kW)": 3.0, "TC (Deg. C)": 1.0, "TO (Deg. C)": 1.0}
    
    # Calculate normalized distances for each parameter
    cols_targets = {"Qo(kW)": qo, "TC (Deg. C)": tc, "TO (Deg. C)": to}
    
    # Calculate standard deviations for normalization (avoid division by zero)
    sigmas = {}
    for col in cols_targets.keys():
        std_val = sub[col].std()
        sigmas[col] = std_val if std_val > 0.01 else 1.0  # Minimum std to avoid division by very small numbers
    
    print(f"Standard deviations - Qo: {sigmas['Qo(kW)']:.2f}, TC: {sigmas['TC (Deg. C)']:.2f}, TO: {sigmas['TO (Deg. C)']:.2f}")
    
    # Calculate weighted distance
    total_distance = np.zeros(len(sub))
    
    for col, target in cols_targets.items():
        # Normalized difference
        normalized_diff = (sub[col] - target) / sigmas[col]
        weighted_diff = weights[col] * (normalized_diff ** 2)
        total_distance += weighted_diff
    
    # Add distance as square root (Euclidean distance)
    sub = sub.copy()
    sub["distance"] = np.sqrt(total_distance)
    
    # Calculate absolute differences for display
    sub["ΔQo"] = (sub["Qo(kW)"] - qo).round(3)
    sub["ΔTC"] = (sub["TC (Deg. C)"] - tc).round(3)
    sub["ΔTO"] = (sub["TO (Deg. C)"] - to).round(3)
    sub["Qo_diff_percent"] = (abs(sub["ΔQo"]) / qo * 100).round(1)
    
    # Sort by distance (closest first), then by COP (highest first) as tiebreaker
    sub = sub.sort_values(["distance", "COP"], ascending=[True, False])
    
    # Select output columns
    out_cols = REQUIRED_COLS + ["ΔQo", "ΔTC", "ΔTO", "Qo_diff_percent", "distance"]
    result = sub.head(k)[out_cols]
    
    print(f"Returning top {len(result)} matches")
    if len(result) > 0:
        print(f"Best match: Qo={result.iloc[0]['Qo(kW)']} kW (Δ={result.iloc[0]['ΔQo']} kW, {result.iloc[0]['Qo_diff_percent']}% diff)")
    
    return result

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
            return jsonify({
                'success': False, 
                'message': f'No heat pumps found for refrigerant "{refrigerant}" with the specified criteria. Try adjusting your search parameters or relax exact temperature matches.',
                'available_refrigerants': sorted(df["Refrigerant"].dropna().unique().tolist())
            })
        
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
