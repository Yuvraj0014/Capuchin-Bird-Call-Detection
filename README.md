# 🎵 Capuchin Bird Call Detection System

## 🦜 About the Capuchin Bird

The Capuchin Bird (Perissocephalus tricolor), also known as the Calfbird, is a fascinating species found in the Amazon rainforest. This unique bird is named after the Capuchin monks due to its distinctive bare head that resembles the monks' brown habits.

### Key Characteristics:
- **Size**: Medium-sized bird, approximately 33 cm (13 inches) in length
- **Appearance**: Distinguished by its bare, dark gray head and neck, with olive-brown plumage
- **Habitat**: Primarily inhabits humid lowland forests in South America, particularly in the Amazon Basin
- **Distribution**: Found in countries including Brazil, French Guiana, Suriname, and Venezuela
- **Vocalization**: Known for its remarkable calls that sound similar to a distant cow's moo or a deep horn-like sound
- **Behavior**: Generally solitary birds, most active during dawn and dusk
- **Diet**: Primarily frugivorous, feeding on various forest fruits and berries

## 🔍 Project Overview

This project implements a deep learning-based system for detecting Capuchin bird calls in audio recordings. The system utilizes advanced audio processing techniques and neural networks to identify the unique vocalizations of these fascinating birds.

### 🛠️ Technical Architecture

1. **Audio Processing Pipeline**:
   - Loads audio files using librosa library
   - Converts audio into mel spectrograms for analysis
   - Preprocesses data to match the model's input requirements
   - Utilizes a trained deep learning model for classification

2. **User Interface**:
   - Built with Streamlit for an intuitive, web-based experience
   - Supports multiple audio file uploads
   - Provides real-time processing and feedback
   - Displays confidence scores for detections

### 💻 How It Works

1. **Audio Input**:
   - Users can upload multiple audio files (WAV or MP3 format)
   - Files are temporarily stored in an 'uploads' directory
   - Each file is processed individually

2. **Processing Steps**:
   ```python
   # Audio preprocessing
   - Load audio file using librosa
   - Generate mel spectrogram
   - Resize to target shape (1491, 257)
   - Prepare for model input
   ```

3. **Model Prediction**:
   - Preprocessed audio is fed into the trained model
   - Model generates confidence scores
   - Scores above 50% indicate positive detection

4. **Results Display**:
   - Audio player for each uploaded file
   - Visual confidence indicators
   - Color-coded results (green for detected, red for not detected)
   - Summary of all processed files

### 🎯 Output Interpretation

The system provides results in several formats:

1. **Individual File Results**:
   ```
   Processing file: example.wav
   CAPUCHIN BIRD CALL FOUND (Confidence: 85.32%)
   ```

2. **Summary Report**:
   ```
   📊 Summary of Results
   example1.wav: Capuchin Bird Call Found with 85.32% confidence
   example2.wav: No Capuchin Bird Call Found with 92.45% confidence
   ```

### 🚀 Getting Started

1. **Prerequisites**:
   ```bash
   pip install streamlit librosa numpy keras tensorflow
   ```

2. **Running the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Using the Interface**:
   - Navigate to the provided local URL
   - Upload audio files using the file uploader
   - Wait for processing and view results

### 🎓 Use Cases

- **Wildlife Research**: Monitor Capuchin bird populations and behavior
- **Environmental Studies**: Track species distribution and habitat preferences
- **Conservation**: Assist in biodiversity assessment and protection efforts
- **Educational**: Teaching tool for bird call recognition and bioacoustics

## 📊 Model Performance

The current model achieves:
- High accuracy in detecting Capuchin bird calls
- Low false positive rate
- Robust performance across various audio qualities
- Real-time processing capabilities

## 🔮 Future Scope

### 1. Technical Enhancements
- **Model Improvements**:
  - Integration of more advanced deep learning architectures
  - Implementation of transfer learning techniques
  - Enhanced noise reduction algorithms
  - Multi-species detection capabilities

- **Real-time Processing**:
  - Live audio stream analysis
  - Mobile application development
  - Edge device deployment
  - Low-latency processing optimization

### 2. Feature Expansions
- **Advanced Analytics**:
  - Temporal pattern analysis
  - Geographical distribution mapping
  - Population density estimation
  - Behavioral pattern recognition

- **User Interface Enhancements**:
  - Interactive spectrogram visualization
  - Customizable detection thresholds
  - Batch processing capabilities
  - Mobile-responsive design

### 3. Research Applications
- **Ecological Studies**:
  - Integration with climate data
  - Habitat monitoring systems
  - Migration pattern tracking
  - Ecosystem health assessment

### 4. Integration & Deployment
- **Hardware Integration**:
  - Support for specialized recording devices
  - IoT device compatibility
  - Autonomous recording units

- **Cloud Services**:
  - Scalable cloud deployment
  - API development
  - Database integration
  - Automated backup systems

### 5. Conservation Tools
- **Monitoring Systems**:
  - Automated population tracking
  - Habitat disturbance detection
  - Conservation effectiveness assessment
  - Early warning systems


## 🙏 Acknowledgments

- Thanks to the wildlife recording community for providing training data
- Powered by TensorFlow and Streamlit


