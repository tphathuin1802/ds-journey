import cv2

# Tải Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mở camera
camera_id = 0  # 0 là camera mặc định
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera")
        break

    # Chuyển frame sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ hình chữ nhật xung quanh khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Detected Faces", frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
