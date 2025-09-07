from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image
import json
import os

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Using YOLOv8 nano version for lightweight detection

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'ai_backend'})

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Get image from request
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            image = Image.open(file.stream)
        else:
            # Base64 encoded image
            image_data = request.json['image_data']
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Perform object detection
        results = model(image_cv)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = box.cls[0].cpu().numpy().astype(int)
                class_name = model.names[class_id]
                
                detections.append({
                    'class': class_name,
                    'class_id': int(class_id),
                    'confidence': float(confidence),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                })
        
        # Draw bounding boxes on image
        output_image = image_cv.copy()
        for detection in detections:
            bbox = detection['bbox']
            cv2.rectangle(output_image, 
                         (int(bbox['x1']), int(bbox['y1'])), 
                         (int(bbox['x2']), int(bbox['y2'])), 
                         (0, 255, 0), 2)
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            cv2.putText(output_image, label, 
                       (int(bbox['x1']), int(bbox['y1'])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert output image to base64
        _, buffer = cv2.imencode('.jpg', output_image)
        output_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'detections': detections,
            'output_image': output_image_base64,
            'total_objects': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
