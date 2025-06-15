from flask import Flask, render_template, request
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

app = Flask(__name__)


def get_player_features(player_name):
    try:
        response = requests.get(f"http://localhost:5000/players/{player_name}", timeout=10)
        
        if response.status_code != 200:
            print(f"API Error ({response.status_code}) for {player_name}")
            return None
            
        data = response.json()
        if not data:
            print(f"No data found for {player_name}")
            return None
            
        # main data of player from returned json
        player_data = data[0] if isinstance(data, list) else data
        
        # Helper function to safely extract nested data
        def safe_get(data_dict, keys, default=0):
            current = data_dict
            for key in keys:   #loops through the keys(each return a dictionary), then returns the final value which is not a dictionary
                if isinstance(current, dict):
                    current = current.get(key, default)
                else:
                    return default
            return current or default
        

        features = {
            'name': player_data.get('name', player_name),
            'role': player_data.get('role', 'Unknown'),

            # t20 Features
            't20_avg': float(safe_get(player_data, ['batting_stats', 't20', 'average'], 0)),
            't20_sr': float(safe_get(player_data, ['batting_stats', 't20', 'strike_rate'], 0)),
            't20_wickets': int(safe_get(player_data, ['bowling_stats', 't20', 'wickets'], 0)),

            # odi Features
            'odi_avg': float(safe_get(player_data, ['batting_stats', 'odi', 'average'], 0)),
            'odi_sr': float(safe_get(player_data, ['batting_stats', 'odi', 'strike_rate'], 0)),

            # test Features
            'test_sr': float(safe_get(player_data, ['batting_stats', 'test', 'strike_rate'], 0)),
            'test_avg': float(safe_get(player_data, ['batting_stats', 'test', 'average'], 0)),
            
            'centuries': int(safe_get(player_data, ['batting_stats', 't20', 'hundreds'], 0)) +
                        int(safe_get(player_data, ['batting_stats', 'odi', 'hundreds'], 0)) +
                        int(safe_get(player_data, ['batting_stats', 'test', 'hundreds'], 0))
        }
        
        # target score calculate (customize weights as needed)
        features['score'] = (
            features['t20_avg'] * 0.5 +
            features['t20_sr'] * 0.3 +
            features['t20_wickets'] * 2 +
            features['centuries'] * 1
        )
        
        return features
        
    except Exception as e:
        print(f"Error processing {player_name}: {str(e)}")
        return None

# model training
def train_model(player_names):
    dataset = []
    
    for name in player_names:
        if features := get_player_features(name):
            dataset.append(features)
    
    if not dataset:
        raise ValueError("No valid player data found. Check API and input names.")
    
    df = pd.DataFrame(dataset)
    
    required_cols = ['t20_avg', 't20_sr', 't20_wickets', 'odi_avg', 'odi_sr', 'test_avg', 'centuries']
    

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
            
    
    X = df[required_cols]
    y = df['score']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    dump(model, 'player_model.pkl')
    df.to_csv('player_data.csv', index=False)
    
    return model



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        player_names = [name.strip() for name in request.form['players'].split(',') if name.strip()]
        
        try:
            model = load('player_model.pkl')
        except:
            model = train_model(player_names)
            
        predictions = []
        for name in player_names:
            if features := get_player_features(name):
                X = pd.DataFrame([[
                    features['test_avg'],
                    features['odi_avg'],
                    features['t20_avg'],
                    features['test_sr'],
                    features['odi_sr'],
                    features['t20_sr'],
                    features['centuries']
                ]])
                
                predictions.append({
                    'name': name,
                    'score': round(model.predict(X)[0], 2)
                })
        
        top_players = sorted(predictions, key=lambda x: x['score'], reverse=True)[:11]
        return render_template('index.html', players=top_players)
    
    return render_template('index.html', players=[])

if __name__ == '__main__':
    # Initial training with verified players
    verified_players = [
        "Virat Kohli", "MS Dhoni", "Rohit Sharma", "Shubman Gill", "Suryakumar Yadav", "KL Rahul", "Hardik Pandya", "Rishabh Pant", "Ravindra Jadeja", "Jasprit Bumrah", "Farokh Engineer", "Adam Gilchrist", "Shane Warne", "Ricky Ponting", "Brian Lara", "Muttiah Muralitharan", "AB de Villiers", "Steve Smith", "David Warner", "Joe Root", "Ben Stokes"
    ]
    
    try:
        train_model(verified_players)
    except Exception as e:
        print(f"Initial training failed: {str(e)}")
        exit(1)
        
    app.run(port=8080, debug=True)
