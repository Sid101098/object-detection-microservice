from flask import Flask, request, jsonify, render_template
import requests
import base64
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# AI service endpoint
AI_SERVICE_URL = "http://ai_backend:5001/detect"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image file
        image_data = file.read()
        
        # Send to AI service
        files = {'image': (file.filename, image_data, file.content_type)}
        response = requests.post(AI_SERVICE_URL, files=files)
        
        if response.status_code != 200:
            return jsonify({'error': 'AI service error', 'details': response.json()}), 500
        
        result = response.json()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON result
        json_filename = f"detection_{timestamp}.json"
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        with open(json_path, 'w') as f:
            f.write(response.text)
        
        # Save output image
        if 'output_image' in result:
            image_data = base64.b64decode(result['output_image'])
            image_filename = f"output_{timestamp}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            result['output_image_url'] = f"/static/uploads/{image_filename}"
        
        result['json_url'] = f"/static/uploads/{json_filename}"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'ui_backend'})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
