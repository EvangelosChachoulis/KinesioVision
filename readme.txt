# Kinesio-Vision: AI Squat Tracker & Database

Kinesio-Vision is a Computer Vision application designed to analyze an athlete's biomechanics during a bodyweight squat. The system dynamically calculates knee joint angles, counts repetitions, and evaluates athletic performance. Every session's data is permanently stored in a local SQLite database and can be exported for further statistical analysis. 

## TECH STACK
- Python
- OpenCV
- MediaPipe
- SQLite3
- Pandas
- NumPy

## SETUP
- Clone this repository to your local machine.
- Install the required python dependencies.
- Download the MediaPipe pose landmarker model, and place it in the root directory of the project.
- Open main.py and update the VIDEO_FILE_PATH variable to your testing video.

## HOW TO USE
- Run the script.
- Enter the athlete's name in terminal when prompted.
- The video will launch, displaying real-time pose tracking, the current knee angle, and the live repetition count.
- Press 'q' or ';' to terminate the video processing at any time.
- Performance evaluation will be displayed in terminal. 
- Follow the interactive prompt to export the data.
