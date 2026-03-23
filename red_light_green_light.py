import cv2
import numpy as np
import random
import time

# Game constants
BASE_GREEN_MOVE_THRESHOLD = 0.04
BASE_RED_MOVE_THRESHOLD = 0.055
RED_GRACE_MS = 650
IDLE_WARNING_MS = 1800
IDLE_DEATH_MS = 3600
GREEN_MIN_MS = 2600
GREEN_MAX_MS = 4200
RED_MIN_MS = 1700
RED_MAX_MS = 2900
CYCLES_PER_LEVEL = 3
MAX_LEVEL = 10
DISPLAY_WIDTH = 640

# States
STATE_GREEN = "GREEN"
STATE_RED = "RED"
STATE_WARNING = "WARNING"
STATE_DEAD = "DEAD"

def current_ms():
    return int(time.time() * 1000)

def random_duration(min_ms, max_ms):
    return random.randint(min_ms, max_ms)

def resize_frame(frame, width=DISPLAY_WIDTH):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)

def compute_motion_score(gray, prev_gray):
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(gray, prev_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return float(np.mean(thresh)) / 255.0

def get_level_thresholds(level):
    level = min(MAX_LEVEL, max(1, level))
    green_thresh = max(0.015, BASE_GREEN_MOVE_THRESHOLD - (level - 1) * 0.005)
    red_thresh = BASE_RED_MOVE_THRESHOLD + (level - 1) * 0.007
    return green_thresh, red_thresh

def reset_game():
    return {
        'state': STATE_GREEN,
        'state_start': current_ms(),
        'green_duration': random_duration(GREEN_MIN_MS, GREEN_MAX_MS),
        'red_duration': random_duration(RED_MIN_MS, RED_MAX_MS),
        'idle_start': None,
        'dead_reason': "",
        'dead_since': None,
        'cycles': 0,
        'level': 1
    }

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    game = reset_game()
    prev_gray = None

    print("Red Light Green Light Game")
    print("Controls: 'q' to quit, 'r' to restart after death")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion_score = compute_motion_score(gray, prev_gray)
        prev_gray = gray

        now = current_ms()
        elapsed = now - game['state_start']
        green_thresh, red_thresh = get_level_thresholds(game['level'])

        # Game logic
        if game['state'] in (STATE_GREEN, STATE_WARNING):
            if motion_score < green_thresh:
                if game['idle_start'] is None:
                    game['idle_start'] = now
                idle_time = now - game['idle_start']
            else:
                game['idle_start'] = None
                idle_time = 0
                if game['state'] == STATE_WARNING:
                    game['state'] = STATE_GREEN
                    game['state_start'] = now
                    game['green_duration'] = max(0, game['green_duration'] - elapsed)

            if game['idle_start'] and idle_time > IDLE_WARNING_MS:
                game['state'] = STATE_WARNING

            if game['idle_start'] and idle_time > IDLE_DEATH_MS:
                game['state'] = STATE_DEAD
                game['dead_reason'] = "Idle too long in GREEN"
                game['dead_since'] = now

            if game['state'] != STATE_DEAD and elapsed >= game['green_duration']:
                game['state'] = STATE_RED
                game['state_start'] = now
                game['red_duration'] = random_duration(RED_MIN_MS, RED_MAX_MS)
                game['idle_start'] = None

        elif game['state'] == STATE_RED:
            if elapsed >= RED_GRACE_MS and motion_score > red_thresh:
                game['state'] = STATE_DEAD
                game['dead_reason'] = "Moved during RED"
                game['dead_since'] = now

            if elapsed >= game['red_duration'] and game['state'] != STATE_DEAD:
                game['cycles'] += 1
                if game['cycles'] >= CYCLES_PER_LEVEL:
                    game['cycles'] = 0
                    game['level'] = min(game['level'] + 1, MAX_LEVEL)
                game['state'] = STATE_GREEN
                game['state_start'] = now
                game['green_duration'] = random_duration(GREEN_MIN_MS, GREEN_MAX_MS)
                game['idle_start'] = None

        # UI
        overlay = frame.copy()
        color = (0, 180, 0) if game['state'] in (STATE_GREEN, STATE_WARNING) else (0, 0, 180) if game['state'] == STATE_RED else (0, 0, 200)
        cv2.rectangle(overlay, (0, 0), (DISPLAY_WIDTH, int(frame.shape[0] * 0.14)), color, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        cv2.putText(frame, f"STATE: {game['state']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"LEVEL: {game['level']}  CYCLE: {game['cycles']}/{CYCLES_PER_LEVEL}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MOTION: {motion_score:.3f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"GreenThr: {green_thresh:.3f}  RedThr: {red_thresh:.3f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if game['state'] in (STATE_GREEN, STATE_WARNING):
            green_left = max(0, game['green_duration'] - elapsed)
            idle_time = now - game['idle_start'] if game['idle_start'] else 0
            cv2.putText(frame, f"Next RED: {green_left} ms", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Idle: {idle_time} ms", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if game['state'] == STATE_WARNING:
                cv2.putText(frame, "WARNING: Move or DIE!", (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif game['state'] == STATE_RED:
            red_left = max(0, game['red_duration'] - elapsed)
            grace_left = max(0, RED_GRACE_MS - elapsed)
            cv2.putText(frame, f"Next GREEN: {red_left} ms", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Grace: {grace_left} ms", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif game['state'] == STATE_DEAD:
            cv2.putText(frame, "DEAD! Press 'r' to restart", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, f"Reason: {game['dead_reason']}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Red Light Green Light", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and game['state'] == STATE_DEAD:
            game = reset_game()

        if game['state'] == STATE_DEAD and game.get('dead_since') is not None:
            if now - game['dead_since'] >= 3000:
                game = reset_game()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
