from flask import Flask, Response, render_template, jsonify, request
from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import RPi.GPIO as GPIO
import time
import threading
import cv2
import numpy as np

app = Flask(__name__)

# Connect to Pixhawk
vehicle = None  # Start disconnected

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

# Video Stream Function
PALETTES = [
    None,  # Default: no colormap
    cv2.COLORMAP_AUTUMN,
    cv2.COLORMAP_BONE,
    cv2.COLORMAP_JET,
    cv2.COLORMAP_WINTER,
    cv2.COLORMAP_RAINBOW,
    cv2.COLORMAP_OCEAN,
    cv2.COLORMAP_SUMMER,
    cv2.COLORMAP_SPRING,
    cv2.COLORMAP_COOL,
    cv2.COLORMAP_HSV,
    cv2.COLORMAP_PINK,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_PARULA if hasattr(cv2, "COLORMAP_PARULA") else cv2.COLORMAP_JET,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_CIVIDIS,
    cv2.COLORMAP_TWILIGHT,
    cv2.COLORMAP_TWILIGHT_SHIFTED,
    cv2.COLORMAP_TURBO,
    cv2.COLORMAP_DEEPGREEN if hasattr(cv2, "COLORMAP_DEEPGREEN") else cv2.COLORMAP_SUMMER,
]
PALETTE_NAMES = [
    "Grayscale",
    "Autumn","Bone","Jet","Winter","Rainbow","Ocean","Summer","Spring","Cool",
    "HSV","Pink","Hot","Parula","Magma","Inferno","Plasma","Viridis","Cividis",
    "Twilight","Twilight Shifted","Turbo","Deep Green"
]
current_palette_index = 0
isotherm_enabled = False

def apply_isotherm(gray, lower=75, upper=255):
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = (gray >= lower) & (gray <= upper)
    colored_hot = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    color_img[mask] = colored_hot[mask]
    return color_img

@app.route('/toggle_isotherm', methods=['POST'])
def toggle_isotherm():
    global isotherm_enabled
    isotherm_enabled = not isotherm_enabled
    return jsonify({"success": True, "isotherm": isotherm_enabled})

@app.route('/change_palette', methods=['POST'])
def change_palette():
    global current_palette_index, isotherm_enabled
    current_palette_index = (current_palette_index + 1) % len(PALETTES)
    if PALETTE_NAMES[current_palette_index] != "Grayscale":
        isotherm_enabled = False  # Disable isotherm if not in grayscale
    return jsonify({
        "success": True,
        "palette": PALETTE_NAMES[current_palette_index],
        "isotherm": isotherm_enabled
    })

@app.route('/get_palette_state')
def get_palette_state():
    return jsonify({
        "palette": PALETTE_NAMES[current_palette_index],
        "isotherm": isotherm_enabled
    })


temp_overlay_enabled = False

@app.route('/toggle_temp_overlay', methods=['POST'])
def toggle_temp_overlay():
    global temp_overlay_enabled
    temp_overlay_enabled = not temp_overlay_enabled
    return jsonify({"success": True, "temp_enabled": temp_overlay_enabled})

def pixel_to_celsius(pixel, min_temp=20, max_temp=40):
    return min_temp + (pixel / 255.0) * (max_temp - min_temp)

def draw_temp_overlay(image, gray, min_temp=20, max_temp=40):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    h, w = gray.shape
    center = (w // 2, h // 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    cv2.circle(image, min_loc, 6, (255,0,0), -1)
    cv2.putText(image, f"{pixel_to_celsius(min_val, min_temp, max_temp):.1f}C", (min_loc[0]+8, min_loc[1]), font, font_scale, (255,0,0), thickness, cv2.LINE_AA)
    cv2.circle(image, max_loc, 6, (0,0,255), -1)
    cv2.putText(image, f"{pixel_to_celsius(max_val, min_temp, max_temp):.1f}C", (max_loc[0]+8, max_loc[1]), font, font_scale, (0,0,255), thickness, cv2.LINE_AA)
    ch_x, ch_y = center
    cross_len = 5
    cv2.line(image, (ch_x - cross_len, ch_y), (ch_x + cross_len, ch_y), (0,255,0), 2)
    cv2.line(image, (ch_x, ch_y - cross_len), (ch_x, ch_y + cross_len), (0,255,0), 2)
    center_val = gray[ch_y, ch_x]
    cv2.putText(image, f"{pixel_to_celsius(center_val, min_temp, max_temp):.1f}C", (ch_x+8, ch_y), font, font_scale, (0,255,0), thickness, cv2.LINE_AA)
    return image



def generate_frames():
    global current_palette_index, isotherm_enabled
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Optional: resize to match your UI size
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (256, 192))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        palette = PALETTES[current_palette_index]
        if palette is None:
            # Grayscale mode
            if isotherm_enabled:
                colored = apply_isotherm(gray)
            else:
                colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            # Colormap mode
            colored = cv2.applyColorMap(gray, palette)
        if temp_overlay_enabled:
            colored = draw_temp_overlay(colored, gray)
        _, buffer = cv2.imencode('.jpg', colored)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')






# Motion Sensor Setup
MOTION_SENSOR_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTION_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

motion_detected = False

def monitor_motion():
    global motion_detected
    last_state = GPIO.LOW
    last_detection_time = 0
    DEBOUNCE_DELAY = 3  # seconds

    while True:
        # Only check the sensor if vehicle is disarmed, otherwise clear motion_detected
        if vehicle is not None and getattr(vehicle, "armed", False):
            # Drone is armed: ignore sensor and clear motion_detected
            motion_detected = False
            time.sleep(0.2)  # Sleep to prevent tight loop
            continue

        # Drone is disarmed: sensor works as usual
        current_state = GPIO.input(MOTION_SENSOR_PIN)
        current_time = time.time()

        if current_state == GPIO.HIGH and last_state == GPIO.LOW:
            if (current_time - last_detection_time) > DEBOUNCE_DELAY:
                print("Motion Detected!")
                motion_detected = True
                last_detection_time = current_time

        if (current_time - last_detection_time) > DEBOUNCE_DELAY:
            motion_detected = False

        last_state = current_state
        time.sleep(0.05)

# Start motion sensor thread
motion_thread = threading.Thread(target=monitor_motion, daemon=True)
motion_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/motion_status')
def motion_status():
    return jsonify({"motion": motion_detected})



@app.route('/connect', methods=['POST'])
def connect_pixhawk():
    global vehicle
    if vehicle is not None:
        return jsonify({"status": "already_connected"})
    try:
        vehicle = connect('/dev/ttyAMA0', baud=57600, wait_ready=True)
        return jsonify({"status": "connected"})
    except Exception as e:
        vehicle = None
        return jsonify({"status": "error", "message": str(e)})

@app.route('/connection_status')
def connection_status():
    global vehicle
    is_connected = vehicle is not None
    return jsonify({"connected": is_connected})

@app.route('/disconnect', methods=['POST'])
def disconnect_pixhawk():
    global vehicle
    if vehicle is None:
        return jsonify({"status": "already_disconnected"})
    try:
        vehicle.close()
        vehicle = None
        return jsonify({"status": "disconnected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/status')
def status():
    default_lat = 14.599419
    default_lng = 121.001782

    if vehicle is None:
        return jsonify({"connected": False, "motion": motion_detected})

    try:
        mode = vehicle.mode.name if vehicle.mode else "N/A"

        battery = vehicle.battery
        if battery:
            voltage = f"{battery.voltage:.2f}" if battery.voltage is not None else "N/A"
            current = f"{battery.current:.2f}" if battery.current is not None else "N/A"
            level = f"{battery.level}%" if battery.level is not None else "N/A"
            battery_display = f"{voltage}v, Current: {current} | {level}"
        else:
            battery_display = "N/A"

        lat = getattr(vehicle.location.global_frame, 'lat', None)
        lon = getattr(vehicle.location.global_frame, 'lon', None)
        alt = getattr(vehicle.location.global_relative_frame, 'alt', None)
        gps = f"{lat}, {lon}" if lat is not None and lon is not None else "N/A"

        # Yaw
        import math
        if getattr(vehicle, "attitude", None) and getattr(vehicle.attitude, "yaw", None) is not None:
            yaw_deg = math.degrees(vehicle.attitude.yaw)
            yaw_deg = (yaw_deg + 360) % 360
        else:
            yaw_deg = 0

        # Ground speed (in m/s)
        ground_speed = f"{vehicle.groundspeed:.2f}" if vehicle.groundspeed is not None else "N/A"

        # Vertical speed (vz, in m/s)
        if vehicle.velocity is not None:
            vz = vehicle.velocity[2]  # vz is the vertical speed
            vertical_speed = f"{-vz:.2f}"  # Negative sign so climb is positive
        else:
            vertical_speed = "N/A"

        # Use defaults if lat/lon missing
        resp_lat = lat if lat is not None else default_lat
        resp_lng = lon if lon is not None else default_lng

        return jsonify({
            "connected": True,
            "mode": mode,
            "battery": battery_display,
            "gps": gps,
            "altitude": alt if alt is not None else "N/A",
            "ground_speed": ground_speed,
            "vertical_speed": vertical_speed,
            "armed": vehicle.armed,
            "lat": resp_lat,
            "lng": resp_lng,
            "yaw": yaw_deg,
            "motion": motion_detected
        })
    except Exception as e:
        # If not connected, still return motion!
        return jsonify({
            "connected": False,
            "motion": motion_detected   # <-- Always include motion
        })

@app.route('/arm', methods=['POST'])
def arm_drone():
    if vehicle is None:
        return "Vehicle not connected."
    if not vehicle.armed:
        vehicle.armed = True
        msg = "Drone is arming..."
    else:
        msg = "Drone is already armed."
    print(msg)
    return msg

@app.route('/disarm', methods=['POST'])
def disarm_drone():
    global vehicle
    if vehicle is None:
        return "Vehicle not connected."
    if vehicle.armed:
        vehicle.armed = False
        msg = "Drone is disarming..."
    else:
        msg = "Drone is already disarmed."
    print(msg)
    return msg

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global vehicle
    if vehicle is None:
        return jsonify({'success': False, 'message': 'Vehicle not connected.'})
    mode = request.json.get('mode')
    if not mode:
        return jsonify({'success': False, 'message': 'No mode specified.'})
    try:
        vehicle.mode = VehicleMode(mode.upper())
        return jsonify({'success': True, 'message': f'Mode set to {mode}.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    

@app.route('/takeoff', methods=['POST'])
def takeoff():
    if vehicle is None or not vehicle.is_armable:
        return jsonify({'success': False, 'error': 'Vehicle not armed/ready'})
    try:
        data = request.get_json()
        alt = float(data.get('altitude', 0))
        if alt <= 0:
            return jsonify({'success': False, 'error': 'Invalid altitude'})
        # Only allow in GUIDED mode
        if vehicle.mode.name != "GUIDED":
            return jsonify({'success': False, 'error': 'Not in GUIDED mode'})
        # Arm and takeoff
        vehicle.mode = VehicleMode("GUIDED")
        vehicle.armed = True
        # Wait for arming
        timeout = 10  # seconds
        start_time = time.time()
        while not vehicle.armed and time.time() - start_time < timeout:
            time.sleep(0.5)
        if not vehicle.armed:
            return jsonify({'success': False, 'error': 'Failed to arm'})
        vehicle.simple_takeoff(alt)
        return jsonify({'success': True})
    except Exception as e:
        # Optionally log full traceback here for debugging
        return jsonify({'success': False, 'error': str(e)})


@app.route('/goto', methods=['POST'])
def goto():
    if vehicle is None or not vehicle.is_armable:
        return jsonify({'success': False, 'error': 'Vehicle not ready'})
    try:
        data = request.get_json()
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
        alt = float(data.get('altitude'))
        if vehicle.mode.name != "GUIDED":
            return jsonify({'success': False, 'error': 'Not in GUIDED mode'})
        location = LocationGlobalRelative(lat, lon, alt)
        vehicle.simple_goto(location)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/position')
def position():
    if vehicle is None:
        return jsonify({'success': False, 'error': 'Vehicle not connected'})
    loc = vehicle.location.global_relative_frame
    return jsonify({'success': True,
                    'latitude': loc.lat,
                    'longitude': loc.lon,
                    'altitude': loc.alt})

@app.route('/home_position')
def home_position():
    if vehicle and vehicle.home_location:
        return jsonify({
            'success': True,
            'lat': vehicle.home_location.lat,
            'lon': vehicle.home_location.lon,
            'alt': vehicle.home_location.alt
        })
    else:
        return jsonify({'success': False, 'error': 'Home position not set'})
    


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        GPIO.cleanup()
        cap.release()


