from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Define the path to the results folder
results_folder = 'results'

@app.route('/')
def index():
    locations = [name for name in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, name))]
    return render_template('index.html', locations=locations)

@app.route('/location/<location>')
def get_offense_types(location):
    location_path = os.path.join(results_folder, location)
    offense_types = [name for name in os.listdir(location_path) if os.path.isdir(os.path.join(location_path, name))]
    return jsonify(offense_types)

@app.route('/offense/<location>/<offense_type>')
def get_incidents(location, offense_type):
    offense_path = os.path.join(results_folder, location, offense_type)
    incidents = [name for name in os.listdir(offense_path) if os.path.isdir(os.path.join(offense_path, name))]
    return jsonify(incidents)

@app.route('/incident/<location>/<offense_type>/<incident>')
def incident(location, offense_type, incident):
    return render_template('incident.html', location=location, offense_type=offense_type, incident=incident)

@app.route('/predictions/<location>/<offense_type>/<incident>/<level>', methods=['GET'])
def get_predictions(location, offense_type, incident, level):
    try:
        file_path = None
        incident_path = os.path.join(results_folder, location, offense_type, incident)

        if not os.path.exists(incident_path):
            return jsonify({'error': 'Incident folder not found'}), 404

        for file_name in os.listdir(incident_path):
            if level == 'state' and 'predictions' in file_name and 'states' in file_name:
                file_path = os.path.join(incident_path, file_name)
                break
            elif level == 'university' and 'predictions' in file_name and 'universities' in file_name:
                file_path = os.path.join(incident_path, file_name)
                break
            elif level == 'top_states' and 'top' in file_name and 'states' in file_name:
                file_path = os.path.join(incident_path, file_name)
                break
            elif level == 'top_universities' and 'top' in file_name and 'universities' in file_name:
                file_path = os.path.join(incident_path, file_name)
                break

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        df = pd.read_csv(file_path)
        result = df.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)