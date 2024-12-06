import cv2
import time
import Hand_Tracker_Module as htm


def main():
    prev_time = 0
    cur_time = 0

    cap = cv2.VideoCapture(0)
    hdetect = htm.HandDetector()
    while True:
        success, img = cap.read()
        img = hdetect.findHands(img)
        lm_list = hdetect.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])

        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time

        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
        )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
            break


if __name__ == "__main__":
    main()
