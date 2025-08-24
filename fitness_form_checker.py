import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import math
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import tempfile
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Custom CSS for enhanced UI
def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    
    /* Header styling */
    .stTitle {
        color: white !important;
        text-align: center;
        font-size: 3rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    
    /* Card-like containers */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 5px;
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        color: #667eea;
        font-weight: bold;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
    
    /* Video analysis card */
    .video-analysis-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    /* Success/Error/Warning boxes */
    .stSuccess {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 10px;
        padding: 15px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(76,175,80,0.3);
    }
    
    .stError {
        background-color: #f44336 !important;
        color: white !important;
        border-radius: 10px;
        padding: 15px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(244,67,54,0.3);
    }
    
    .stWarning {
        background-color: #ff9800 !important;
        color: white !important;
        border-radius: 10px;
        padding: 15px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(255,152,0,0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    /* Video container */
    .video-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #6B46C1 0%, #9333EA 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(147,51,234,0.3);
    }
    
    /* Progress bar for video processing */
    .progress-container {
        background: #f0f0f0;
        border-radius: 20px;
        padding: 5px;
        margin: 10px 0;
    }
    
    .progress-bar {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        height: 30px;
        border-radius: 15px;
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    
    /* Analysis results card */
    .analysis-result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

class PoseExtractor:
    """Class to handle MediaPipe pose detection and feature extraction"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def extract_landmarks(self, image):
        """Extract pose landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        return results
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def extract_squat_features(self, landmarks):
        """Extract features specific to squat exercise"""
        if not landmarks.pose_landmarks:
            return None
        
        # Get key landmarks for squat analysis
        points = landmarks.pose_landmarks.landmark
        
        # Hip, knee, ankle points for both legs
        left_hip = [points[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   points[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [points[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    points[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        right_hip = [points[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    points[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Shoulder and spine points
        left_shoulder = [points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
        # Calculate angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Hip angles (using shoulder-hip-knee)
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Spine angle (shoulder to hip vertical alignment)
        spine_angle = abs(left_shoulder[0] - left_hip[0])  # Horizontal deviation
        
        # Knee alignment (knees should track over toes)
        left_knee_alignment = abs(left_knee[0] - left_ankle[0])
        right_knee_alignment = abs(right_knee[0] - right_ankle[0])
        
        # Hip depth (how low the person goes)
        hip_depth = min(left_hip[1], right_hip[1])
        knee_level = min(left_knee[1], right_knee[1])
        squat_depth = hip_depth - knee_level
        
        features = [
            left_knee_angle, right_knee_angle,
            left_hip_angle, right_hip_angle,
            spine_angle,
            left_knee_alignment, right_knee_alignment,
            squat_depth
        ]
        
        return features
    
    def draw_enhanced_pose(self, image, landmarks, prediction, stress_data):
        """Draw enhanced pose visualization with colorful overlays"""
        if not landmarks.pose_landmarks:
            return image
        
        h, w, _ = image.shape
        
        # Create overlay for effects
        overlay = image.copy()
        
        # Add gradient background effect
        gradient = np.zeros_like(overlay)
        for i in range(h):
            gradient[i, :] = [int(255 * (1 - i/h) * 0.2), 
                            int(150 * (1 - i/h) * 0.2), 
                            int(255 * (i/h) * 0.2)]
        overlay = cv2.addWeighted(overlay, 0.7, gradient, 0.3, 0)
        
        # Draw skeleton with custom colors based on stress
        self.draw_colored_skeleton(overlay, landmarks, stress_data, w, h)
        
        # Add form status badge
        status_text = "EXCELLENT FORM!" if prediction == 1 else "NEEDS IMPROVEMENT"
        status_color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
        
        # Badge background
        badge_bg = (0, 200, 0) if prediction == 1 else (200, 0, 0)
        cv2.rectangle(overlay, (w-250, 20), (w-20, 70), badge_bg, -1)
        cv2.rectangle(overlay, (w-250, 20), (w-20, 70), status_color, 3)
        cv2.putText(overlay, status_text, (w-240, 50), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        # Add performance metrics overlay
        self.draw_performance_overlay(overlay, stress_data, w, h)
        
        # Blend with original
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        return result
    
    def draw_colored_skeleton(self, image, landmarks, stress_data, width, height):
        """Draw skeleton with gradient colors based on stress levels"""
        points = landmarks.pose_landmarks.landmark
        
        # Define connections with their stress associations
        connections = [
            # Left leg
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, 'left_knee_stress'),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE, 'left_ankle_stress'),
            # Right leg
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, 'right_knee_stress'),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE, 'right_ankle_stress'),
            # Torso
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP, 'spine_stress'),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP, 'spine_stress'),
            # Arms
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, None),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST, None),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, None),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST, None),
        ]
        
        # Draw connections with gradient effect
        for start, end, stress_key in connections:
            start_point = (int(points[start.value].x * width), 
                          int(points[start.value].y * height))
            end_point = (int(points[end.value].x * width), 
                        int(points[end.value].y * height))
            
            # Get stress level for color
            stress = stress_data.get(stress_key, 0) if stress_key else 0
            
            # Create gradient line
            num_segments = 10
            for i in range(num_segments):
                t1 = i / num_segments
                t2 = (i + 1) / num_segments
                
                x1 = int(start_point[0] * (1 - t1) + end_point[0] * t1)
                y1 = int(start_point[1] * (1 - t1) + end_point[1] * t1)
                x2 = int(start_point[0] * (1 - t2) + end_point[0] * t2)
                y2 = int(start_point[1] * (1 - t2) + end_point[1] * t2)
                
                # Color based on stress
                if stress < 0.3:
                    color = (0, int(255 * (1 - stress)), 0)
                elif stress < 0.7:
                    color = (0, int(255 * (1 - (stress - 0.3) / 0.4)), 
                            int(255 * (stress - 0.3) / 0.4))
                else:
                    color = (int(255 * (stress - 0.7) / 0.3), 0, 
                            int(255 * (1 - (stress - 0.7) / 0.3)))
                
                thickness = int(8 - stress * 3)
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw joints with glow effect
        key_joints = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        
        for joint in key_joints:
            x = int(points[joint.value].x * width)
            y = int(points[joint.value].y * height)
            
            # Glow effect
            for r in range(20, 5, -2):
                alpha = (20 - r) / 15
                color = (int(255 * alpha), int(200 * alpha), int(100 * alpha))
                cv2.circle(image, (x, y), r, color, -1)
            
            # Core joint
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
    
    def draw_performance_overlay(self, image, stress_data, width, height):
        """Draw performance metrics overlay"""
        # Performance bar at bottom
        bar_height = 60
        bar_y = height - bar_height - 20
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (20, bar_y), (width - 20, height - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw stress bars
        metrics = list(stress_data.items())
        bar_width = (width - 40) // len(metrics)
        
        for i, (name, value) in enumerate(metrics):
            x_start = 30 + i * bar_width
            x_end = x_start + bar_width - 10
            
            # Background bar
            cv2.rectangle(image, (x_start, bar_y + 30), (x_end, bar_y + 50), (50, 50, 50), -1)
            
            # Stress bar
            bar_length = int((x_end - x_start) * value)
            if value < 0.3:
                color = (0, 255, 0)
            elif value < 0.7:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(image, (x_start, bar_y + 30), (x_start + bar_length, bar_y + 50), color, -1)
            
            # Label
            label = name.replace('_stress', '').replace('_', ' ').title()
            cv2.putText(image, label, (x_start, bar_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

class SquatFormClassifier:
    """Class to handle squat form classification"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data for squat form"""
        np.random.seed(42)
        
        # Good form characteristics
        good_samples = []
        for _ in range(n_samples // 2):
            # Good squat: knees 90-120°, hips 90-110°, minimal spine deviation
            left_knee = np.random.uniform(90, 120)
            right_knee = np.random.uniform(90, 120)
            left_hip = np.random.uniform(90, 110)
            right_hip = np.random.uniform(90, 110)
            spine = np.random.uniform(0.02, 0.08)  # Minimal deviation
            left_alignment = np.random.uniform(0.02, 0.06)
            right_alignment = np.random.uniform(0.02, 0.06)
            depth = np.random.uniform(0.15, 0.25)  # Good depth
            
            good_samples.append([left_knee, right_knee, left_hip, right_hip, 
                               spine, left_alignment, right_alignment, depth])
        
        # Bad form characteristics
        bad_samples = []
        for _ in range(n_samples // 2):
            # Bad squat: extreme angles, poor alignment
            left_knee = np.random.choice([
                np.random.uniform(60, 89),   # Too shallow
                np.random.uniform(121, 160)  # Too deep/unstable
            ])
            right_knee = np.random.choice([
                np.random.uniform(60, 89),
                np.random.uniform(121, 160)
            ])
            left_hip = np.random.choice([
                np.random.uniform(60, 89),   # Not enough hip hinge
                np.random.uniform(111, 140)  # Too much forward lean
            ])
            right_hip = np.random.choice([
                np.random.uniform(60, 89),
                np.random.uniform(111, 140)
            ])
            spine = np.random.uniform(0.1, 0.3)    # Poor spine alignment
            left_alignment = np.random.uniform(0.08, 0.2)   # Knees cave in/out
            right_alignment = np.random.uniform(0.08, 0.2)
            depth = np.random.choice([
                np.random.uniform(0.05, 0.14),  # Too shallow
                np.random.uniform(0.26, 0.4)    # Too deep
            ])
            
            bad_samples.append([left_knee, right_knee, left_hip, right_hip,
                              spine, left_alignment, right_alignment, depth])
        
        # Create dataset
        X = np.array(good_samples + bad_samples)
        y = np.array([1] * len(good_samples) + [0] * len(bad_samples))  # 1=good, 0=bad
        
        return X, y
    
    def train_model(self):
        """Train the squat form classification model"""
        X, y = self.generate_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return accuracy
    
    def predict_form(self, features):
        """Predict form quality from features"""
        if not self.is_trained or features is None:
            return None, 0.0
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        return prediction, confidence
    
    def get_feedback(self, features):
        """Get detailed feedback based on features"""
        if features is None:
            return "Cannot detect pose properly"
        
        feedback = []
        
        # Check knee angles
        left_knee, right_knee = features[0], features[1]
        if left_knee < 90 or right_knee < 90:
            feedback.append("🎯 Squat deeper - aim for 90° knee bend")
        elif left_knee > 120 or right_knee > 120:
            feedback.append("⚡ Control your descent - avoid going too low")
        
        # Check spine alignment
        spine_deviation = features[4]
        if spine_deviation > 0.1:
            feedback.append("🎯 Keep chest up and spine neutral")
        
        # Check knee alignment
        left_alignment, right_alignment = features[5], features[6]
        if left_alignment > 0.08 or right_alignment > 0.08:
            feedback.append("⚡ Push knees out - track over toes")
        
        # Check squat depth
        depth = features[7]
        if depth < 0.15:
            feedback.append("💪 Go deeper for full muscle activation")
        
        if not feedback:
            feedback = ["🌟 Perfect form! Keep it up!"]
        
        return feedback
    
    def calculate_stress_levels(self, features):
        """Calculate stress levels for different body parts"""
        if features is None:
            return {}
        
        stress_data = {}
        
        # Knee stress based on angle deviation
        left_knee_angle, right_knee_angle = features[0], features[1]
        optimal_knee_min, optimal_knee_max = 90, 120
        
        left_knee_stress = 0
        if left_knee_angle < optimal_knee_min:
            left_knee_stress = (optimal_knee_min - left_knee_angle) / optimal_knee_min
        elif left_knee_angle > optimal_knee_max:
            left_knee_stress = (left_knee_angle - optimal_knee_max) / optimal_knee_max
        
        right_knee_stress = 0
        if right_knee_angle < optimal_knee_min:
            right_knee_stress = (optimal_knee_min - right_knee_angle) / optimal_knee_min
        elif right_knee_angle > optimal_knee_max:
            right_knee_stress = (right_knee_angle - optimal_knee_max) / optimal_knee_max
        
        # Hip stress
        left_hip_angle, right_hip_angle = features[2], features[3]
        optimal_hip_min, optimal_hip_max = 90, 110
        
        left_hip_stress = 0
        if left_hip_angle < optimal_hip_min:
            left_hip_stress = (optimal_hip_min - left_hip_angle) / optimal_hip_min
        elif left_hip_angle > optimal_hip_max:
            left_hip_stress = (left_hip_angle - optimal_hip_max) / optimal_hip_max
        
        right_hip_stress = 0
        if right_hip_angle < optimal_hip_min:
            right_hip_stress = (optimal_hip_min - right_hip_angle) / optimal_hip_min
        elif right_hip_angle > optimal_hip_max:
            right_hip_stress = (right_hip_angle - optimal_hip_max) / optimal_hip_max
        
        # Spine stress
        spine_deviation = features[4]
        spine_stress = min(spine_deviation * 10, 1.0)
        
        # Ankle stress
        left_alignment, right_alignment = features[5], features[6]
        left_ankle_stress = min(left_alignment * 10, 1.0)
        right_ankle_stress = min(right_alignment * 10, 1.0)
        
        stress_data = {
            'left_knee_stress': min(left_knee_stress, 1.0),
            'right_knee_stress': min(right_knee_stress, 1.0),
            'left_hip_stress': min(left_hip_stress, 1.0),
            'right_hip_stress': min(right_hip_stress, 1.0),
            'spine_stress': spine_stress,
            'left_ankle_stress': left_ankle_stress,
            'right_ankle_stress': right_ankle_stress
        }
        
        return stress_data

class VideoAnalyzer:
    """Class to handle video file analysis"""
    
    def __init__(self, pose_extractor, classifier):
        self.pose_extractor = pose_extractor
        self.classifier = classifier
    
    def analyze_video(self, video_path, progress_callback=None):
        """Analyze an entire video and return frame-by-frame results"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = {
            'fps': fps,
            'total_frames': total_frames,
            'frame_data': [],
            'summary': {
                'total_reps': 0,
                'good_form_frames': 0,
                'bad_form_frames': 0,
                'average_knee_angle': 0,
                'average_hip_angle': 0,
                'stress_timeline': []
            }
        }
        
        frame_count = 0
        in_squat = False
        knee_angles = []
        hip_angles = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if progress_callback:
                progress_callback(frame_count / total_frames)
            
            # Extract pose
            landmarks = self.pose_extractor.extract_landmarks(frame)
            features = self.pose_extractor.extract_squat_features(landmarks)
            
            if features:
                # Get prediction
                prediction, confidence = self.classifier.predict_form(features)
                stress_data = self.classifier.calculate_stress_levels(features)
                
                # Track metrics
                knee_angle = (features[0] + features[1]) / 2
                hip_angle = (features[2] + features[3]) / 2
                knee_angles.append(knee_angle)
                hip_angles.append(hip_angle)
                
                # Count reps
                if knee_angle < 100 and not in_squat:
                    in_squat = True
                elif knee_angle > 140 and in_squat:
                    in_squat = False
                    results['summary']['total_reps'] += 1
                
                # Update counters
                if prediction == 1:
                    results['summary']['good_form_frames'] += 1
                else:
                    results['summary']['bad_form_frames'] += 1
                
                # Store frame data
                frame_info = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'prediction': prediction,
                    'confidence': confidence,
                    'features': features,
                    'stress_data': stress_data,
                    'feedback': self.classifier.get_feedback(features)
                }
                results['frame_data'].append(frame_info)
                
                # Store stress timeline
                avg_stress = np.mean(list(stress_data.values()))
                results['summary']['stress_timeline'].append(avg_stress)
            
            frame_count += 1
        
        # Calculate averages
        if knee_angles:
            results['summary']['average_knee_angle'] = np.mean(knee_angles)
        if hip_angles:
            results['summary']['average_hip_angle'] = np.mean(hip_angles)
        
        cap.release()
        return results
    
    def create_highlight_reel(self, video_path, results, output_path):
        """Create a highlight video with annotations"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        for frame_info in results['frame_data']:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count == frame_info['frame_number']:
                # Extract pose for visualization
                landmarks = self.pose_extractor.extract_landmarks(frame)
                
                # Draw enhanced visualization
                frame_with_viz = self.pose_extractor.draw_enhanced_pose(
                    frame, landmarks, 
                    frame_info['prediction'], 
                    frame_info['stress_data']
                )
                
                # Add timestamp
                timestamp_text = f"Time: {frame_info['timestamp']:.1f}s"
                cv2.putText(frame_with_viz, timestamp_text, (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame_with_viz)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        return output_path

def create_animated_stress_gauge(stress_value, label):
    """Create an animated gauge chart for stress visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = stress_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': label, 'font': {'size': 20}},
        delta = {'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': 'lightgreen'},
                {'range': [0.3, 0.7], 'color': 'yellow'},
                {'range': [0.7, 1], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_rep_counter_display(rep_count, target_reps=10):
    """Create a visual rep counter with progress bar"""
    progress = min(rep_count / target_reps * 100, 100)
    
    return f"""
    <div style="text-align: center; margin: 20px 0;">
        <h2 style="color: #667eea; margin-bottom: 10px;">Rep Counter</h2>
        <div style="font-size: 48px; font-weight: bold; color: #764ba2; margin: 10px 0;">
            {rep_count} / {target_reps}
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress}%;">
                {int(progress)}%
            </div>
        </div>
    </div>
    """

def create_video_analysis_summary(results):
    """Create a comprehensive summary of video analysis"""
    total_time = results['total_frames'] / results['fps']
    good_form_percentage = (results['summary']['good_form_frames'] / 
                           results['total_frames'] * 100) if results['total_frames'] > 0 else 0
    
    # Create timeline chart
    fig_timeline = go.Figure()
    
    # Add stress timeline
    timestamps = [i/results['fps'] for i in range(len(results['summary']['stress_timeline']))]
    fig_timeline.add_trace(go.Scatter(
        x=timestamps,
        y=results['summary']['stress_timeline'],
        mode='lines',
        name='Average Stress',
        line=dict(color='rgb(255, 0, 0)', width=2)
            ))
    
    # Add form quality indicators
    good_form_times = []
    bad_form_times = []
    for frame_info in results['frame_data']:
        if frame_info['prediction'] == 1:
            good_form_times.append(frame_info['timestamp'])
        else:
            bad_form_times.append(frame_info['timestamp'])
    
    if good_form_times:
        fig_timeline.add_trace(go.Scatter(
            x=good_form_times,
            y=[0.1] * len(good_form_times),
            mode='markers',
            name='Good Form',
            marker=dict(color='green', size=8, symbol='circle')
        ))
    
    if bad_form_times:
        fig_timeline.add_trace(go.Scatter(
            x=bad_form_times,
            y=[0.9] * len(bad_form_times),
            mode='markers',
            name='Needs Improvement',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig_timeline.update_layout(
        title="Form Quality Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Stress Level / Form Indicators",
        height=300,
        showlegend=True,
        yaxis=dict(range=[0, 1])
    )
    
    return {
        'total_time': total_time,
        'good_form_percentage': good_form_percentage,
        'timeline_chart': fig_timeline
    }

def render_live_camera_tab(form_checker):
    """Render the live camera analysis tab"""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
            <h2 style="color: #667eea; text-align: center;">📹 Live Exercise Feed</h2>
        </div>
        """, unsafe_allow_html=True)
        
        video_placeholder = st.empty()
        
        # Camera control
        camera_on = st.checkbox("🔴 Enable Live Camera", key="live_camera_toggle")
        
        if camera_on and st.session_state.trained:
            # Start camera and process feed
            pose_extractor = form_checker.pose_extractor
            
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                st.error("❌ Cannot access camera")
            else:
                # Create placeholder for live updates
                feedback_placeholder = col2.empty()
                
                while camera_on and st.session_state.get('live_camera_toggle', False):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break
                    
                    # Extract pose
                    results = pose_extractor.extract_landmarks(frame)
                    
                    # Extract features
                    features = pose_extractor.extract_squat_features(results)
                    
                    # Get prediction
                    prediction, confidence = form_checker.classifier.predict_form(features)
                    
                    # Get feedback
                    feedback_list = form_checker.classifier.get_feedback(features)
                    
                    # Calculate stress levels
                    stress_data = form_checker.classifier.calculate_stress_levels(features)
                    
                    # Draw enhanced visualization
                    frame_with_viz = pose_extractor.draw_enhanced_pose(
                        frame, results, prediction, stress_data)
                    
                    # Display frame
                    video_placeholder.image(frame_with_viz, channels="BGR", use_column_width=True)
                    
                    # Update feedback panel
                    render_feedback_panel(feedback_placeholder, prediction, confidence, 
                                        features, stress_data, feedback_list)
                
                cap.release()
        
        elif camera_on and not st.session_state.trained:
            st.markdown("""
            <div style="background: #FFF3CD; color: #856404; padding: 20px; 
                       border-radius: 15px; text-align: center;">
                <h3>⚠️ Please train the model first!</h3>
                <p>Click the "Train Model" button in the sidebar to get started.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Show welcome screen
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       color: white; padding: 40px; border-radius: 20px; text-align: center;">
                <h2 style="font-size: 2.5rem; margin-bottom: 20px;">Live Camera Analysis 🎯</h2>
                <p style="font-size: 1.2rem; margin-bottom: 30px;">
                    Get real-time feedback on your form as you exercise!
                </p>
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px;">
                    <h3>Instructions:</h3>
                    <ol style="text-align: left; font-size: 1.1rem;">
                        <li>Enable your camera using the checkbox above</li>
                        <li>Position yourself so your full body is visible</li>
                        <li>Start performing squats</li>
                        <li>Watch for real-time feedback on your form</li>
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if not camera_on:
            render_exercise_guide()

def render_video_upload_tab(form_checker):
    """Render the video upload and analysis tab"""
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
        <h2 style="color: #667eea; text-align: center;">📁 Video File Analysis</h2>
        <p style="text-align: center; color: #666;">Upload your workout video for detailed analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video of yourself performing squats"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video info
        st.markdown(f"""
        <div class="uploadedFile">
            <h4>📹 Uploaded: {uploaded_file.name}</h4>
            <p>Size: {uploaded_file.size / 1024 / 1024:.2f} MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis options
        col1, col2, col3 = st.columns(3)
        with col1:
            analyze_full = st.checkbox("Full Analysis", value=True)
        with col2:
            create_highlights = st.checkbox("Create Highlights", value=True)
        with col3:
            export_report = st.checkbox("Export Report", value=True)
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not st.session_state.trained:
                st.error("⚠️ Please train the model first!")
            else:
                # Create progress container
                progress_container = st.container()
                with progress_container:
                    st.markdown("""
                    <div class="analysis-result-card">
                        <h3 style="text-align: center;">🔍 Analyzing Video...</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize video analyzer
                    video_analyzer = VideoAnalyzer(
                        form_checker.pose_extractor,
                        form_checker.classifier
                    )
                    
                    # Progress callback
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {progress*100:.1f}%")
                    
                    # Analyze video
                    with st.spinner("Analyzing video..."):
                        results = video_analyzer.analyze_video(video_path, update_progress)
                    
                    # Clear progress
                    progress_container.empty()
                    
                    # Display results
                    display_video_analysis_results(results, video_path, video_analyzer, 
                                                 create_highlights, export_report)
        
        # Clean up temp file
        try:
            os.unlink(video_path)
        except:
            pass
    else:
        # Show upload instructions
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                   color: white; padding: 40px; border-radius: 20px; text-align: center; margin-top: 20px;">
            <h2 style="font-size: 2rem; margin-bottom: 20px;">Upload Your Workout Video 📹</h2>
            <p style="font-size: 1.1rem; margin-bottom: 20px;">
                Get comprehensive analysis of your exercise form from recorded videos
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px; text-align: left;">
                <h4>What you'll get:</h4>
                <ul style="font-size: 1rem;">
                    <li>✅ Frame-by-frame form analysis</li>
                    <li>📊 Detailed performance metrics</li>
                    <li>🎯 Stress level visualization</li>
                    <li>💡 Personalized improvement tips</li>
                    <li>🎬 Annotated highlight reel</li>
                    <li>📄 Exportable analysis report</li>
                </ul>
            </div>
            <p style="margin-top: 20px; font-size: 0.9rem;">
                Supported formats: MP4, AVI, MOV, MKV
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_video_analysis_results(results, video_path, video_analyzer, 
                                   create_highlights, export_report):
    """Display comprehensive video analysis results"""
    # Summary metrics
    st.markdown("""
    <div class="analysis-result-card">
        <h2 style="text-align: center; margin-bottom: 30px;">📊 Analysis Complete!</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    summary = results['summary']
    total_time = results['total_frames'] / results['fps']
    good_form_percentage = (summary['good_form_frames'] / 
                           results['total_frames'] * 100) if results['total_frames'] > 0 else 0
    
    with col1:
        st.metric("Total Reps", summary['total_reps'], 
                 help="Number of complete squats detected")
    
    with col2:
        st.metric("Good Form %", f"{good_form_percentage:.1f}%",
                 delta=f"{good_form_percentage-50:.1f}%" if good_form_percentage > 50 else None,
                 help="Percentage of frames with good form")
    
    with col3:
        st.metric("Avg Knee Angle", f"{summary['average_knee_angle']:.1f}°",
                 help="Average knee angle during squats")
    
    with col4:
        st.metric("Duration", f"{total_time:.1f}s",
                 help="Total video duration")
    
    # Timeline visualization
    st.markdown("### 📈 Performance Timeline")
    summary_data = create_video_analysis_summary(results)
    st.plotly_chart(summary_data['timeline_chart'], use_container_width=True)
    
    # Detailed frame analysis
    with st.expander("🔍 Detailed Frame Analysis"):
        # Sample frames for visualization
        sample_indices = np.linspace(0, len(results['frame_data'])-1, 
                                   min(10, len(results['frame_data']))).astype(int)
        
        st.markdown("#### Sample Frames Analysis")
        frame_cols = st.columns(5)
        
        for idx, frame_idx in enumerate(sample_indices[:5]):
            frame_info = results['frame_data'][frame_idx]
            with frame_cols[idx % 5]:
                if frame_info['prediction'] == 1:
                    st.success(f"Frame {frame_info['frame_number']}")
                else:
                    st.error(f"Frame {frame_info['frame_number']}")
                st.caption(f"Time: {frame_info['timestamp']:.1f}s")
    
    # Stress analysis
    st.markdown("### 🎯 Body Part Stress Analysis")
    
    # Calculate average stress for each body part
    avg_stress = {}
    for part in ['left_knee_stress', 'right_knee_stress', 'left_hip_stress', 
                 'right_hip_stress', 'spine_stress', 'left_ankle_stress', 'right_ankle_stress']:
        stress_values = [frame['stress_data'][part] for frame in results['frame_data'] 
                        if frame['stress_data']]
        if stress_values:
            avg_stress[part] = np.mean(stress_values)
    
    # Display stress gauges
    gauge_cols = st.columns(4)
    for idx, (part, stress) in enumerate(avg_stress.items()):
        with gauge_cols[idx % 4]:
            part_name = part.replace('_stress', '').replace('_', ' ').title()
            fig = create_animated_stress_gauge(stress, part_name)
            st.plotly_chart(fig, use_container_width=True)
    
    # Feedback summary
    st.markdown("### 💡 Common Issues Detected")
    
    # Aggregate feedback
    feedback_counts = {}
    for frame_info in results['frame_data']:
        if frame_info['feedback']:
            for feedback in frame_info['feedback']:
                feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
    
    # Sort by frequency
    sorted_feedback = sorted(feedback_counts.items(), key=lambda x: x[1], reverse=True)
    
    for feedback, count in sorted_feedback[:5]:
        frequency = count / len(results['frame_data']) * 100
        if frequency > 20:
            st.warning(f"{feedback} (Detected in {frequency:.0f}% of frames)")
        else:
            st.info(f"{feedback} (Detected in {frequency:.0f}% of frames)")
    
    # Create highlights if requested
    if create_highlights:
        st.markdown("### 🎬 Creating Highlight Reel...")
        with st.spinner("Generating annotated video..."):
            output_path = "highlighted_video.mp4"
            video_analyzer.create_highlight_reel(video_path, results, output_path)
            
            # Provide download link
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=video_bytes,
                    file_name="squat_analysis_highlights.mp4",
                    mime="video/mp4"
                )
            
            # Clean up
            try:
                os.unlink(output_path)
            except:
                pass
    
    # Export report if requested
    if export_report:
        st.markdown("### 📄 Export Analysis Report")
        report = generate_analysis_report(results)
        st.download_button(
            label="📥 Download Report (PDF)",
            data=report,
            file_name="squat_analysis_report.pdf",
            mime="application/pdf"
        )

def render_feedback_panel(placeholder, prediction, confidence, features, stress_data, feedback_list):
    """Render the feedback panel with all metrics"""
    with placeholder.container():
        # Form quality indicator
        if prediction is not None:
            if prediction == 1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%); 
                            color: white; padding: 20px; border-radius: 15px; 
                            text-align: center; margin-bottom: 20px;">
                    <h2 style="margin: 0;">✨ EXCELLENT FORM!</h2>
                    <p style="margin: 5px 0; font-size: 1.2rem;">Keep it up! 💪</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f44336 0%, #FF6B6B 100%); 
                            color: white; padding: 20px; border-radius: 15px; 
                            text-align: center; margin-bottom: 20px;">
                    <h2 style="margin: 0;">⚠️ NEEDS IMPROVEMENT</h2>
                    <p style="margin: 5px 0; font-size: 1.2rem;">Check the tips below 👇</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown(f"""
            <div style="background: #f0f0f0; border-radius: 10px; padding: 10px; margin-bottom: 20px;">
                <h4 style="margin: 0; color: #667eea;">AI Confidence: {confidence:.0%}</h4>
                <div style="background: #ddd; border-radius: 5px; height: 20px; margin-top: 5px;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               width: {confidence*100}%; height: 100%; border-radius: 5px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Rep counter update
            if features:
                # Simple rep counting logic based on knee angle
                knee_angle = (features[0] + features[1]) / 2
                if knee_angle < 100 and not st.session_state.in_squat:
                    st.session_state.in_squat = True
                elif knee_angle > 140 and st.session_state.in_squat:
                    st.session_state.in_squat = False
                    st.session_state.rep_count += 1
                    st.session_state.session_stats['total_reps'] += 1
                    if prediction == 1:
                        st.session_state.session_stats['good_form_reps'] += 1
            
            # Display rep counter
            st.markdown(create_rep_counter_display(st.session_state.rep_count), 
                       unsafe_allow_html=True)
            
            # Feedback tips
            st.markdown("""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #667eea; margin-bottom: 10px;">💡 Form Tips</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for tip in feedback_list:
                st.info(tip)
            
            # Pose metrics
            if features:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #6B46C1 0%, #9333EA 100%); 
                           color: white; padding: 20px; border-radius: 15px; margin-top: 20px;">
                    <h3 style="text-align: center; margin-bottom: 15px;">📊 Pose Metrics</h3>
                </div>
                """, unsafe_allow_html=True)
                
                metric_cols = st.columns(2)
                metrics = [
                    ("Left Knee", features[0], "°"),
                    ("Right Knee", features[1], "°"),
                    ("Left Hip", features[2], "°"),
                    ("Right Hip", features[3], "°"),
                    ("Spine Align", features[4] * 100, "%"),
                    ("Depth", features[7] * 100, "%")
                ]
                
                for idx, (name, value, unit) in enumerate(metrics):
                    col = metric_cols[idx % 2]
                    with col:
                        # Determine if value is in good range
                        is_good = True
                        if "Knee" in name or "Hip" in name:
                            is_good = 90 <= value <= 120
                        elif "Align" in name:
                            is_good = value < 8
                        elif "Depth" in name:
                            is_good = 15 <= value <= 25
                        
                        color = "#4CAF50" if is_good else "#f44336"
                        
                        st.markdown(f"""
                        <div style="background: {color}; color: white; padding: 10px; 
                                   border-radius: 10px; margin: 5px 0; text-align: center;">
                            <b>{name}</b><br>
                            <span style="font-size: 1.5rem;">{value:.1f}{unit}</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("🔍 No pose detected - make sure you're in frame!")

def render_exercise_guide():
    """Render exercise guide and tips"""
    st.markdown("""
    <div style="background: white; padding: 30px; border-radius: 20px; 
               box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
        <h2 style="color: #667eea; text-align: center; margin-bottom: 20px;">
            🏋️‍♂️ Perfect Squat Guide
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Exercise tips with colorful cards
    tips = [
        ("🦵", "Stance", "Feet shoulder-width apart, toes slightly outward"),
        ("📐", "Depth", "Lower until thighs are parallel to ground"),
        ("🎯", "Knees", "Keep knees tracking over toes"),
        ("🏔️", "Back", "Maintain neutral spine throughout"),
        ("⚖️", "Weight", "Keep weight balanced on mid-foot"),
        ("💨", "Breathing", "Inhale down, exhale up")
    ]
    
    for icon, title, desc in tips:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                   color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h4 style="margin: 0;">{icon} {title}</h4>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

def generate_analysis_report(results):
    """Generate a PDF report of the analysis (placeholder for actual PDF generation)"""
    # This is a placeholder - in a real implementation, you would use a library like reportlab
    # to generate an actual PDF report
    report_content = f"""
    SQUAT FORM ANALYSIS REPORT
    ==========================
    
    Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    SUMMARY METRICS
    ---------------
    Total Reps: {results['summary']['total_reps']}
    Good Form Frames: {results['summary']['good_form_frames']}
    Bad Form Frames: {results['summary']['bad_form_frames']}
    Good Form Percentage: {(results['summary']['good_form_frames'] / results['total_frames'] * 100):.1f}%
    
    Average Knee Angle: {results['summary']['average_knee_angle']:.1f}°
    Average Hip Angle: {results['summary']['average_hip_angle']:.1f}°
    
    RECOMMENDATIONS
    ---------------
    Based on the analysis, focus on:
    1. Maintaining consistent knee angle between 90-120 degrees
    2. Keeping spine neutral throughout the movement
    3. Ensuring knees track over toes
    
    For detailed frame-by-frame analysis, please refer to the highlighted video.
    """
    
    return report_content.encode('utf-8')

def create_streamlit_app():
    """Create enhanced Streamlit interface with tabs"""
    st.set_page_config(
        page_title="AI Fitness Form Checker", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Header with gradient background
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🏋️‍♂️ AI-Powered Fitness Form Checker
        </h1>
        <p style="color: white; font-size: 1.2rem; margin-top: 10px;">
            Real-time squat form analysis using advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'form_checker' not in st.session_state:
        st.session_state.form_checker = FitnessFormChecker()
        st.session_state.trained = False
        st.session_state.rep_count = 0
        st.session_state.in_squat = False
        st.session_state.session_stats = {
            'total_reps': 0,
            'good_form_reps': 0,
            'start_time': None
        }
    
    # Sidebar with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h2 style="color: white; text-align: center; margin: 0;">Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Training section
        st.markdown("### 🎯 Step 1: Train AI Model")
        col1, col2 = st.columns([3, 1])
        with col1:
            train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
        with col2:
            if st.session_state.trained:
                st.success("✅")
        
        if train_button:
            with st.spinner("🧠 Training AI model..."):
                accuracy = st.session_state.form_checker.train_system()
                st.session_state.trained = True
                st.success(f"✨ Model trained! Accuracy: {accuracy:.1%}")
                st.balloons()
        
        st.markdown("---")
        
        # Session stats
        if st.session_state.session_stats['total_reps'] > 0:
            st.markdown("### 📊 Session Statistics")
            st.metric("Total Reps", st.session_state.session_stats['total_reps'])
            good_form_percentage = (st.session_state.session_stats['good_form_reps'] / 
                                  st.session_state.session_stats['total_reps'] * 100)
            st.metric("Good Form %", f"{good_form_percentage:.1f}%")
        
        # Reset button
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.rep_count = 0
            st.session_state.session_stats = {
                'total_reps': 0,
                'good_form_reps': 0,
                'start_time': None
            }
            st.success("Session reset!")
        
        # Help section
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **Quick Tips:**
            - Train the model first (one-time setup)
            - Choose between live camera or video upload
            - Ensure good lighting and full body visibility
            - Stand 6-10 feet from camera
            
            **Troubleshooting:**
            - Camera not working? Check browser permissions
            - Low accuracy? Ensure proper lighting
            - Slow performance? Close other applications
            """)
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["📹 Live Camera", "📁 Upload Video"])
    
    with tab1:
        render_live_camera_tab(st.session_state.form_checker)
    
    with tab2:
        render_video_upload_tab(st.session_state.form_checker)

class FitnessFormChecker:
    """Main application class"""
    
    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.classifier = SquatFormClassifier()
    
    def train_system(self):
        """Train the form classification system"""
        return self.classifier.train_model()

# Custom footer
def add_footer():
    st.markdown("""
    <div style="text-align: center; padding: 30px; margin-top: 50px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px;">
        <p style="color: white; margin: 0;">
            Built with ❤️ using Streamlit, MediaPipe, and Machine Learning<br>
            <small>Stay fit, stay healthy! 💪</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_streamlit_app()
    add_footer()
