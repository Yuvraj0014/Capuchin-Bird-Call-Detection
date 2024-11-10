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

### 🚀 Getting Started with Installation

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

The Capuchin Bird Call Detection System has significant potential for expansion and enhancement in multiple directions. Future developments could include real-time audio processing capabilities for live monitoring, integration with mobile applications for field researchers, and support for multi-species bird call detection to broaden its ecological impact. The system could be enhanced with advanced machine learning techniques such as transfer learning and automated model retraining to improve accuracy and reduce false positives. Integration with IoT devices and autonomous recording units would enable widespread deployment in remote locations, while the addition of geographical mapping and temporal analysis features would provide valuable insights for conservation efforts. Looking ahead, the project aims to evolve into a comprehensive ecosystem monitoring tool by incorporating climate data analysis, population density estimation, and habitat health assessment capabilities, making it an invaluable resource for wildlife researchers, conservationists, and environmental scientists.

## 🙏 Acknowledgments

- Thanks to the wildlife recording community for providing training data
- Powered by TensorFlow and Streamlit


