import cv2
import numpy as np
import datetime

def detect_waterbottle(video_path):
    
    # open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    
    water_bottle_detected = False  # indicate whether water bottle is detected
    
    while True:
        # read a frame from video
        ret, frame = cap.read()
        
        # ret = true (frame successful)
        if ret:
            # Convert frame to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # lower and upper color bounds for blue
            lower_blue = np.array([100, 100, 50])
            upper_blue = np.array([140, 255, 255])
            
            # frame threshold
            mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
            
            # contours in blue mask
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # if other contours found
            if contours:
                # get contour with largest area (water bottle)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # get bounding rectangle of contour
                x, y, w, h = cv2.boundingRect(largest_contour)

                # draw rectangle around water bottle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if (150 < x < 350) and (100 < y < 300):
                    if not water_bottle_detected:
                        print("Water bottle detected in specified area!")
                        water_bottle_detected = True  # Set the flag to True
                        
                        cv2.putText(frame, "Capturing photo...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
                        
                        for i in range(3, 0, -1):
                            frame_filled = np.zeros_like(frame)
                            
                            # countdown
                            cv2.putText(frame_filled, str(i), (50 + i * 50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)
                            
                            # combine original and filled frame
                            frame_display = cv2.addWeighted(frame, 0.7, frame_filled, 0.3, 0)
                            
                            # display frame
                            cv2.imshow('Water Bottle Detection', frame_display)
                            cv2.waitKey(1000)  

                        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        filename = f'detected_waterbottle_{timestamp}.jpg'
                        
                        # save frame as img
                        cv2.imwrite(filename, frame)
                        
                        #
                        captured_img = cv2.imread(filename)
                        cv2.imshow("Captured img", captured_img)
                        cv2.waitKey(0)
                        
                        
                        break  
          
            cv2.imshow('Water Bottle Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_waterbottle(0)  
