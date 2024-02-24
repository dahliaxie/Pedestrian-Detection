import cv2

# Load the pre-trained pedestrian detection model
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Function to estimate pedestrian dimensions from the first frame
def estimate_pedestrian_dimensions(frame):
    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect pedestrians in the frame
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
    # If no pedestrians are detected, return default dimensions
    if len(pedestrians) == 0:
        return (80, 200)  # Default width and height
    # Otherwise, estimate dimensions based on the first detected pedestrian
    (x, y, w, h) = pedestrians[0]
    return (w, h)

# Function to detect pedestrians in a frame
def detect_pedestrians(frame, min_pedestrian_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adjust parameters for better detection performance
    pedestrians = pedestrian_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,          # Adjust scaleFactor based on the trade-off between detection rate and speed
        minNeighbors=1,           # Adjust minNeighbors based on false positives and missed detections
        minSize=min_pedestrian_size  # Adjust minSize based on the typical size of individual pedestrians
    )
    return pedestrians

# Function to draw bounding boxes around pedestrians and add labels
def draw_boxes(frame, pedestrians):
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        # Calculate the count of pedestrians within the current bounding box
        pedestrian_count = sum(x <= (x2 + w2) and (x + w) >= x2 and y <= (y2 + h2) and (y + h) >= y2 for (x2, y2, w2, h2) in pedestrians)
        # Add label with the count of pedestrians detected within the current bounding box
        cv2.putText(frame, f'{pedestrian_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

# Main function for pedestrian detection and tracking
def main():
    cap = cv2.VideoCapture('peoplebg.mp4')  # Load video file
    
    # Estimate pedestrian dimensions from the first frame
    _, sample_frame = cap.read()
    min_pedestrian_size = estimate_pedestrian_dimensions(sample_frame)
    print(min_pedestrian_size)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        pedestrians = detect_pedestrians(frame, min_pedestrian_size)
        frame_with_boxes = draw_boxes(frame, pedestrians)
        
        cv2.imshow('Pedestrian Detection', frame_with_boxes)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
