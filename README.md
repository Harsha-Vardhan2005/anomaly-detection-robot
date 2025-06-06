# Security System with Autonomous Anomaly Detection and Response Robot

![Security Robot Banner](results/system_architecture.jpg)

An intelligent autonomous security robot system that combines real-time anomaly detection with robotic response mechanisms, deployed on NVIDIA Jetson Nano. The system utilizes advanced AI technologies including YOLOv8, DeepSort, MediaPipe Pose, MiDaS depth estimation, and LLaMA integration for comprehensive security monitoring and response.

## 🎯 Key Features

- **Real-time Anomaly Detection**: YOLOv8-powered detection of suspicious objects (weapons, abandoned baggage)
- **Multi-Object Tracking**: DeepSort algorithm for robust person and object tracking with unique ID assignment
- **Behavioral Analysis**: MediaPipe Pose estimation for detecting loitering and suspicious movements
- **Emotion Recognition**: CNN-based facial emotion detection for threat assessment
- **Autonomous Navigation**: ROS-powered SLAM navigation with LiDAR sensors
- **Intelligent Interaction**: LLaMA 3.1-based conversational AI for suspect interrogation
- **Depth Perception**: MiDaS monocular depth estimation for spatial awareness
- **Edge Computing**: Optimized for NVIDIA Jetson Nano platform

## 🏗️ System Architecture

![System Architecture](results/methodology.jpg)

The system integrates multiple AI components:
- **Vision Pipeline**: Multi-camera surveillance with YOLOv8 object detection
- **Tracking Module**: DeepSort for persistent object and person tracking
- **Behavioral Analysis**: MediaPipe Pose for movement pattern analysis
- **Navigation System**: ROS-based autonomous robot navigation with SLAM
- **Interaction Module**: Natural language processing for human-robot interaction

## 🛠️ Technology Stack

### AI/ML Components
- **YOLOv8m**: Fine-tuned object detection (80% mAP)
- **DeepSort**: Multi-object tracking with Kalman filtering
- **MediaPipe Pose**: Real-time human pose estimation
- **MiDaS**: Monocular depth estimation
- **CNN**: Emotion recognition (85% accuracy on FER-2013)
- **LLaMA 3.1**: 8B parameter language model for conversations

### Hardware & Platforms
- **NVIDIA Jetson Nano**: Edge computing platform
- **LiDAR Sensors**: Environmental mapping and navigation
- **Multi-Camera Setup**: Comprehensive area surveillance
- **ROS (Robot Operating System)**: Robotics framework

### 📂 Datasets
## 🔍 1. Suspicious Activity Detection Dataset

- **Source**: Roboflow
- **Link**: [Suspicious Detection Dataset](https://universe.roboflow.com/suspicious-detection/suspicious-detection-rnkj7/dataset/1)
- **Usage**: Used for training an anomaly detection model to identify suspicious behavior.

---

## 😊 2. Facial Emotion Recognition (FER2013)

- **Source**: Kaggle
- **Link**: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Usage**: Used to train an emotion detection model for analyzing people's emotions post-detection.

---

💡 **After Downloading**:
- Extract the dataset folders
- Place the extracted folders inside this `data/` directory


### Libraries & Dependencies
```bash
torch>=1.9.0
ultralytics>=8.0.0
opencv-python>=4.5.0
mediapipe>=0.8.10
numpy>=1.21.0
scipy>=1.7.0
transformers>=4.21.0
rospy>=1.15.0
```

## 🚀 Quick Start

### Prerequisites
- NVIDIA Jetson Nano (4GB recommended)
- Python 3.8+
- ROS Melodic/Noetic
- CUDA 10.2+
- Multiple USB cameras or IP cameras
- LiDAR sensor (compatible with ROS)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/security-robot-system.git
cd security-robot-system
```

2. **Setup Jetson Nano environment**
```bash
chmod +x scripts/setup/setup_jetson.sh
./scripts/setup/setup_jetson.sh
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
chmod +x scripts/setup/install_dependencies.sh
./scripts/setup/install_dependencies.sh
```

4. **Setup ROS workspace**
```bash
cd ros_workspace
catkin_make
source devel/setup.bash
```

5. **Configure cameras and calibration**
```bash
python scripts/setup/calibrate_cameras.py
```

6. **Download pre-trained models**
```bash
# Download models from releases or train your own
python scripts/training/train_yolo.py  # For custom training
```

## 🎮 Usage

### Basic System Launch
```bash
# Launch the complete security system
python scripts/deployment/run_system.py

# Or use ROS launch file
roslaunch security_robot security_system.launch
```

### Component Testing
```bash
# Test individual components
python scripts/deployment/test_components.py --component detection
python scripts/deployment/test_components.py --component navigation
python scripts/deployment/test_components.py --component interaction
```

### Configuration
Edit configuration files in the `config/` directory:
```yaml
# config/system_config.yaml
detection:
  confidence_threshold: 0.5
  suspicious_objects: ["knife", "gun", "abandoned_bag"]
  
navigation:
  max_speed: 1.0
  obstacle_distance: 0.5
  
interaction:
  conversation_timeout: 60
  threat_keywords: ["help", "emergency", "weapon"]
```

## 📊 Performance Results

### Detection Performance
![Detection Results](results/yolo.jpg)

- **YOLOv8m mAP**: 80% on custom suspicious object dataset
- **Real-time FPS**: 15-20 FPS on Jetson Nano
- **Detection Classes**: Weapons, people, bags, suspicious objects

### Tracking Accuracy

- **DeepSort Tracking**: 92% accuracy in multi-object scenarios
- **ID Consistency**: 95% ID retention across occlusions
- **Loitering Detection**: 10-second threshold with 90% accuracy

### Emotion Recognition
![Emotion Detection](results/emotion-detection1.jpg)

- **CNN Model Accuracy**: 85% on FER-2013 dataset
- **Real-time Processing**: 10 FPS emotion analysis
- **Emotion Classes**: Angry, Fear, Happy, Neutral, Sad, Surprise, Disgust

### Navigation Performance

- **SLAM Mapping**: Real-time environment mapping
- **Path Planning**: Optimal path calculation in <2 seconds
- **Obstacle Avoidance**: 95% success rate in complex environments

### Human-Robot Interaction
![Interrogation Demo](results/interrogation.jpg)

- **LLaMA Integration**: Context-aware security conversations
- **Response Time**: <3 seconds for generated responses
- **Threat Assessment**: 88% accuracy in suspicious behavior classification

### Alert System
![Interrogation Demo](results/alerts.jpg)

- **Alerts**: Captures and gives the real time alert notifications of suspicious objects
- **Platform**: via telegram app
- **Details**: Gives snapshot, co-ordinates w.r.t camera, weapon id

## 🔧 Advanced Configuration

### Camera Setup
```python
# src/core/camera_manager.py
camera_config = {
    'camera_1': {
        'position': [0, 0, 2.5],
        'rotation': [0, -30, 0],
        'resolution': [1920, 1080],
        'fps': 30
    },
    'camera_2': {
        'position': [10, 0, 2.5],
        'rotation': [0, -30, 180],
        'resolution': [1920, 1080],
        'fps': 30
    }
}
```

### Model Fine-tuning
```python
# Train custom YOLOv8 model
from src.training.train_yolo import YOLOTrainer

trainer = YOLOTrainer(
    dataset_path='data/datasets/suspicious_detection',
    model_size='medium',
    epochs=100,
    batch_size=16
)
trainer.train()
```

### ROS Integration
```python
# ROS node for anomaly detection
import rospy
from security_robot.msg import Anomaly

def anomaly_callback(msg):
    if msg.threat_level > 0.7:
        # Deploy robot for investigation
        navigate_to_coordinates(msg.x, msg.y, msg.z)
        
rospy.Subscriber('/anomaly_detection', Anomaly, anomaly_callback)
```

## 📈 System Workflow

1. **Multi-Camera Surveillance**: Continuous monitoring of designated areas
2. **Object Detection**: YOLOv8 identifies suspicious objects and people
3. **Tracking**: DeepSort assigns unique IDs and tracks movement patterns
4. **Behavioral Analysis**: MediaPipe analyzes poses for suspicious behavior
5. **Anomaly Trigger**: System detects loitering, weapons, or abandoned objects
6. **Robot Deployment**: Autonomous navigation to anomaly location
7. **Human Interaction**: LLaMA-powered conversation for threat assessment
8. **Decision Making**: AI determines threat level and appropriate response
9. **Alert System**: Security personnel notification if necessary

## 🔒 Security Features

- **Real-time Threat Detection**: Multiple AI models for comprehensive analysis
- **Autonomous Response**: No human operator required for initial assessment
- **Privacy Protection**: Local processing on Jetson Nano, no cloud dependency
- **Scalable Architecture**: Support for multiple robots and camera networks
- **Audit Trail**: Complete logging of all detections and interactions

## 🚧 Future Enhancements

- **Multi-Robot Coordination**: Swarm intelligence for large area coverage
- **6G Communication**: Enhanced real-time coordination capabilities
- **Advanced Gesture Recognition**: Non-verbal threat assessment
- **Weather Resistance**: Outdoor deployment capabilities
- **Mobile App Integration**: Remote monitoring and control interface

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This security system is designed for research and educational purposes. Always ensure compliance with local laws and regulations regarding surveillance and privacy. The system should complement, not replace, human security personnel for critical security applications.


⭐ If you find this project helpful, please give it a star!
