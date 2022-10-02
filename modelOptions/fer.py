import cv2
from modelOptions.fer import FER

# Input Image
img1_path = './images/good1.jpeg'
image = cv2.imread(img1_path)
emotion_detector = FER()
# Output image's information
print(emotion_detector.detect_emotions(image))
result = emotion_detector.detect_emotions(image)
emotions = result[0]["emotions"]
emotion_name, score = emotion_detector.top_emotion(image)

#TODO: can use these emotions to filter pictures too
