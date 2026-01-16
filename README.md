

# 🎵 Music Recommendation Based on Facial Expression

## Overview

This project explores an end-to-end **computer vision–driven recommendation system** that maps **real-time facial expressions** to music recommendations. The goal is to study how affective signals extracted from visual data can be used as implicit user feedback for personalized media recommendation.

The system performs **real-time facial emotion recognition** using a webcam feed and recommends music playlists aligned with the detected emotional state.



## Problem Statement

Traditional music recommendation systems rely heavily on:

* Explicit user input (likes, skips, ratings), or
* Historical listening behavior

These approaches fail in scenarios where:

* The user is new (cold start),
* The user cannot or does not want to provide explicit feedback,
* Emotional context matters more than long-term preferences.

This project investigates whether **facial expressions**, captured passively and in real time, can serve as a proxy for user mood and enable **context-aware music recommendations**.



## Key Contributions

* Built a **real-time emotion recognition pipeline** using computer vision.
* Integrated facial emotion output with a **rule-based music recommendation layer**.
* Designed a modular pipeline separating:

  * Face detection
  * Emotion inference
  * Recommendation logic
* Analyzed system limitations and real-world failure cases.



## System Architecture

1. **Video Capture**

   * Live webcam feed using OpenCV.

2. **Face Detection**

   * Haar Cascade / CNN-based face detector (depending on configuration).

3. **Emotion Recognition**

   * Facial emotion classification model (via `fer` library).
   * Output classes include:
     `happy, sad, angry, neutral, fear, surprise, disgust`.

4. **Recommendation Engine**

   * Emotion → mood mapping.
   * Mood → predefined music playlist mapping.

5. **Output**

   * Recommended music playlist displayed to the user.



## Technologies Used

* **Python**
* **OpenCV** – video capture and face detection
* **FER** – facial emotion recognition
* **NumPy / Pandas** – data handling
* **YouTube playlists** – music source



## Experimental Setup

* Input: Live webcam video stream.
* Environment: Indoor lighting, frontal face orientation.
* Inference: Real-time emotion prediction per frame.
* Recommendation update: Triggered on dominant emotion over a short time window.



## Results and Observations

* The system successfully detects facial expressions in real time under controlled conditions.
* Emotion prediction is reasonably stable for **clear expressions** such as happiness and sadness.
* Neutral and mixed expressions show higher ambiguity.
* Music recommendations align well with dominant emotional states in most cases.

*(Note: Quantitative accuracy depends on the pre-trained FER model and was not retrained in this implementation.)*



## Limitations

* Emotion recognition accuracy drops under:

  * Poor lighting
  * Occlusions (mask, hand on face)
  * Extreme head pose
* Emotion-to-music mapping is **rule-based**, not learned.
* No user feedback loop to adapt recommendations over time.
* Emotion recognition model is pre-trained; no dataset-specific fine-tuning was performed.



## Future Work

* Replace rule-based recommendation with a **learning-based recommender**.
* Incorporate **temporal emotion smoothing** using sequence models.
* Add user feedback to refine recommendations.
* Extend to multi-modal inputs (facial expression + audio + context).
* Fine-tune emotion recognition model on a curated dataset.



## Why This Project Matters

This project demonstrates:

* Practical application of **computer vision in human-centered systems**.
* Integration of perception models with downstream decision logic.
* Understanding of real-time constraints and system limitations.

It serves as a foundation for more advanced research in:

* Affective computing
* Context-aware recommendation systems
* Human–AI interaction


