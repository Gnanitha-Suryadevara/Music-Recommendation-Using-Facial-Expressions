# Importing the required libraries
import cv2
from keras.models import load_model # type: ignore
import numpy as np
from youtubesearchpython import VideosSearch # Import YouTube search library
import webbrowser # Import webbrowser to open links
import os
import time # Import time for delays, tracking, and getting the year
import random # Import the random module
import json # Import JSON for persistence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
MODEL_PATH = "model/fer2013_mini_XCEPTION.102-0.66.hdf5" # Make sure this path is correct
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_CONFIRMATION_DURATION = 1 # How long to see the emotion before reacting (adjust if needed)
EMOTION_COOLDOWN_PERIOD = 30 # Seconds before playing again for the SAME emotion in one session
WINDOW_NAME = 'Facial Expression Recognition - Press Q to Quit'
YOUTUBE_SEARCH_LIMIT = 20 # How many results to fetch for random selection
PERSISTENCE_FILE = "last_played_songs.json" # Stores last played URL per emotion
AVAILABLE_LANGUAGES = ["Telugu", "Hindi", "English"] # Languages to choose from randomly

# --- Film Industry Mapping ---
FILM_INDUSTRIES = {
    "Telugu": "Tollywood",
    "Hindi": "Bollywood",
    "English": "Hollywood"
}

# --- Emotion to Song Type Mapping ---
emotion_song_mapping = {
    "Happy": ["happy song", "upbeat song", "feel good song"],
    "Sad": ["sad song", "emotional song", "heartbreak song"],
    "Angry": ["powerful song", "energetic song", "intense song"],
    "Neutral": ["melodious song", "soothing song", "peaceful song"],
    "Surprise": ["upbeat song", "catchy song", "popular song"],
    "Fear": ["calm song", "soothing song", "relaxing song"],
    "Disgust": ["intense song", "heavy beat song", "powerful song"]
}

# --- List of Simple Search Terms by Industry ---
INDUSTRY_SEARCH_TERMS = {
    "Telugu": ["Telugu movie songs", "Tollywood hits", "Telugu film songs"],
    "Hindi": ["Bollywood songs", "Hindi movie songs", "Hindi film songs"],
    "English": ["Hollywood songs", "English movie songs", "Film soundtrack"]
}

# --- Minimum Video Duration (to filter out shorts) ---
MIN_VIDEO_DURATION_SECONDS = 60  # Minimum 1 minute to avoid shorts

# --- Load Model ---
# Ensure the model file exists at MODEL_PATH
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure the path is correct and the model file exists.")
    exit()
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please ensure Keras/TensorFlow is installed correctly.")
    exit()

# --- Check Haar Cascade ---
if not os.path.exists(HAAR_CASCADE_PATH):
    print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
    print("Please ensure OpenCV is installed correctly.")
    exit()
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# --- Emotions List ---
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Persistence Functions ---
def load_last_played():
    """Loads the last played URLs from the JSON file."""
    try:
        with open(PERSISTENCE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"'{PERSISTENCE_FILE}' not found or invalid. Starting fresh.")
        return {}

def save_last_played(data):
    """Saves the last played URLs dictionary to the JSON file."""
    try:
        with open(PERSISTENCE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving last played songs to {PERSISTENCE_FILE}: {e}")

# --- State Variables ---
current_tracking_emotion = None
emotion_start_time = None
last_played_time_for_emotion = {} # For in-session cooldown
last_played_urls = load_last_played() # Load persistent data at start

# --- Function to extract duration in seconds from YouTube duration string ---
def parse_duration(duration_str):
    """Convert YouTube duration string to seconds"""
    try:
        # Handle format like "3:45" (minutes:seconds)
        if ':' in duration_str and duration_str.count(':') == 1:
            minutes, seconds = duration_str.split(':')
            return int(minutes) * 60 + int(seconds)
        # Handle format like "1:23:45" (hours:minutes:seconds)
        elif ':' in duration_str and duration_str.count(':') == 2:
            hours, minutes, seconds = duration_str.split(':')
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        # Handle single number (seconds only)
        else:
            return int(duration_str)
    except (ValueError, TypeError):
        # Return a default value if parsing fails
        return 0

# --- Function to Search and Open YouTube Video ---
def play_video(final_emotion):
    global last_played_urls, last_played_time_for_emotion

    final_emotion = final_emotion.capitalize()
    
    # --- Choose Song Type Based on Emotion ---
    if final_emotion not in emotion_song_mapping:
        print(f"No song type mapped for emotion: {final_emotion}")
        return False
        
    # Pick a random song type for this emotion
    song_types = emotion_song_mapping[final_emotion]
    song_type = random.choice(song_types)
    
    # --- Select Language/Industry ---
    # Randomly select a language
    selected_language = random.choice(AVAILABLE_LANGUAGES)
    industry = FILM_INDUSTRIES[selected_language]
    
    # Select a random search term for this industry
    industry_term = random.choice(INDUSTRY_SEARCH_TERMS[selected_language])
    
    # --- Build Simple Search Query ---
    # Create a much simpler query that will return better results
    search_query = f"{industry_term} {song_type}"
    
    # 50% chance to add "full song" to the query
    if random.random() > 0.5:
        search_query += " full song"
    
    # 25% chance to add the year
    if random.random() > 0.75:
        year = random.randint(2010, 2025)  # Random year between 2010-2025
        search_query += f" {year}"
    
    print(f"\nStable emotion confirmed: {final_emotion}.")
    print(f"Searching for {industry} songs...")
    print(f"Query: '{search_query}'")

    try:
        videosSearch = VideosSearch(search_query, limit=YOUTUBE_SEARCH_LIMIT)
        results = videosSearch.result()

        if not results or not results['result']:
            print(f"No YouTube videos found for query: '{search_query}'")
            # Try an even simpler query as fallback
            fallback_query = f"{industry} {song_type}"
            print(f"Trying fallback query: '{fallback_query}'")
            
            videosSearch = VideosSearch(fallback_query, limit=YOUTUBE_SEARCH_LIMIT)
            results = videosSearch.result()
            
            if not results or not results['result']:
                print(f"No YouTube videos found for fallback query either.")
                return False

        video_list = results['result']
        last_url_played_for_this_emotion = last_played_urls.get(final_emotion)

        # Filter out shorts and previously played
        suitable_videos = []
        for video in video_list:
            # Skip if this isn't a video or is the last played video
            if video.get('type') != 'video' or video.get('link') == last_url_played_for_this_emotion:
                continue
                
            # Parse duration and filter out shorts (< MIN_VIDEO_DURATION_SECONDS)
            duration_str = video.get('duration', '0:00')
            duration_seconds = parse_duration(duration_str)
            
            # Must be a decent length video (not a short)
            if duration_seconds < MIN_VIDEO_DURATION_SECONDS:
                continue
                
            # Contains negative keywords we want to exclude
            title = video.get('title', '').lower()
            negative_keywords = ['short', '#short', 'shorts', '#shorts', 'tiktok', 'teaser', 'trailer']
            if any(keyword in title for keyword in negative_keywords):
                continue
                
            # This video passes all filters
            suitable_videos.append(video)

        if not suitable_videos:
            print("No suitable videos found matching criteria.")
            return False

        # Select a random video from the suitable ones
        random_video = random.choice(suitable_videos)
        video_url = random_video['link']
        video_title = random_video.get('title', 'N/A')
        video_channel = random_video.get('channel', {}).get('name', 'Unknown Channel')
        video_duration = random_video.get('duration', 'Unknown duration')

        print(f"Opening: '{video_title}' by {video_channel}")
        print(f"Duration: {video_duration}")
        print(f"URL: {video_url}")
        webbrowser.open(video_url)
        print("Video opened in browser.")

        # --- Update Persistence and Cooldown ---
        current_time = time.time()
        last_played_urls[final_emotion] = video_url # Update persistent record
        save_last_played(last_played_urls)          # Save to file
        last_played_time_for_emotion[final_emotion] = current_time # Update in-session cooldown timer

        return True

    except Exception as e:
        print(f"Error searching or opening YouTube video: {e}")
        return False


# --- Setup Video Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device (camera).")
    print("Please ensure a camera is connected and drivers are installed.")
    exit()

print(f"Starting facial expression recognition. Runs continuously.")
print(f"Observing for {EMOTION_CONFIRMATION_DURATION}s stable emotion. Cooldown: {EMOTION_COOLDOWN_PERIOD}s.")
print(f"Searching for film songs from {', '.join([FILM_INDUSTRIES[lang] for lang in AVAILABLE_LANGUAGES])}.")
print("Look at the camera. Press 'q' to quit.")

# --- Main Loop (Runs Continuously) ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        time.sleep(0.5)
        continue

    frame = cv2.flip(frame, 1) # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    detected_emotion_in_frame = None

    # --- Process Detected Faces ---
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0: continue

        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        try:
            prediction = model.predict(roi, verbose=0)
            emotion_index = np.argmax(prediction[0])
            detected_emotion_in_frame = emotions[emotion_index]

            # Draw rectangle and text on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion_in_frame, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            break # Process first face only for simplicity
        except Exception as e:
            # print(f"Error during model prediction: {e}") # Optional debug print
            continue

    # --- Emotion Stability and Cooldown Logic ---
    current_time = time.time()

    if detected_emotion_in_frame:
        if detected_emotion_in_frame == current_tracking_emotion:
            # Emotion is consistent, check duration
            if emotion_start_time and (current_time - emotion_start_time >= EMOTION_CONFIRMATION_DURATION):
                # Stable emotion detected, check in-session cooldown
                last_played_this_session = last_played_time_for_emotion.get(current_tracking_emotion, 0)
                if (current_time - last_played_this_session > EMOTION_COOLDOWN_PERIOD):
                    # Cooldown passed for this session, attempt to play
                    if play_video(current_tracking_emotion):
                        # Reset tracking state after successful play
                        emotion_start_time = None
                        current_tracking_emotion = None
                        print(f"Cooldown started for {detected_emotion_in_frame}. Waiting {EMOTION_COOLDOWN_PERIOD}s.")
                    else:
                        # If playing failed, reset tracking anyway
                        current_tracking_emotion = None
                        emotion_start_time = None
                        print("Failed to play video. Resetting emotion tracking.")
                # else: Cooldown active for this session, do nothing

        else:
            # New or changed emotion detected, start/reset tracking
            if detected_emotion_in_frame != current_tracking_emotion:
                print(f"Tracking new emotion: {detected_emotion_in_frame}...")
            current_tracking_emotion = detected_emotion_in_frame
            emotion_start_time = current_time
    else:
        # No face/emotion detected, reset tracking
        if current_tracking_emotion is not None:
             print("Face lost or emotion unclear. Resetting tracking.")
        current_tracking_emotion = None
        emotion_start_time = None

    # --- Display Tracking Status ---
    status_text = "Status: Waiting..."
    if current_tracking_emotion and emotion_start_time:
        elapsed = current_time - emotion_start_time
        status_text = f"Status: Tracking '{current_tracking_emotion}' for {elapsed:.1f}s / {EMOTION_CONFIRMATION_DURATION}s"
        # Check if cooldown is active for the currently tracked emotion
        last_played_this_session = last_played_time_for_emotion.get(current_tracking_emotion, 0)
        if (current_time - last_played_this_session <= EMOTION_COOLDOWN_PERIOD):
             cooldown_remaining = EMOTION_COOLDOWN_PERIOD - (current_time - last_played_this_session)
             status_text += f" (Cooldown: {cooldown_remaining:.0f}s)"

    cv2.putText(frame, status_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Display Frame ---
    cv2.imshow(WINDOW_NAME, frame)

    # --- Check for Quit Key ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting manually...")
        break

# --- Release Resources ---
cap.release()
cv2.destroyAllWindows()
print("Application finished.")