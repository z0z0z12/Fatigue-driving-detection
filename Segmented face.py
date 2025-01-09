import cv2
import dlib
import os


def extract_face(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img_gray)
    for face in faces:
        x, y = face.left(), face.top()
        w = face.width()
        h = face.height()
        img_height, img_width = img_gray.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x_end = min(img_width, x + w)
        y_end = min(img_height, y + h)
        return image[y: y_end, x: x_end]
    return None


def main():
    dir_path = "data"
    train_data_path = os.path.join(dir_path, "validation_data")
    output_base_path = os.path.join("data1", "validation_data")

    for pa in os.listdir(train_data_path):
        sub_path = os.path.join(train_data_path, pa)
        output_sub_path = os.path.join(output_base_path, pa)
        os.makedirs(output_sub_path, exist_ok=True)
        count = 0

        for img in os.listdir(sub_path):
            image_path = os.path.join(sub_path, img)
            try:
                imag = cv2.imread(image_path)
                if imag is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                imag1 = extract_face(imag)

                if imag1 is not None:
                    count += 1
                    print(f"Processed {count} images for {pa}")
                    output_file_name = os.path.join(output_sub_path, f"{count}_{img.split('.')[0]}_{pa}.jpg")
                    cv2.imwrite(output_file_name, imag1)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")


if __name__ == "__main__":
    main()



