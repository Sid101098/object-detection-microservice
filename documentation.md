# Object Detection Microservice Documentation

## Overview
This microservice provides object detection capabilities through a web interface. It consists of two main components:
1. **UI Backend**: Handles file uploads and serves the web interface
2. **AI Backend**: Performs object detection using YOLOv8

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- (Optional) NVIDIA GPU with CUDA for accelerated inference

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd object-detection-microservice
