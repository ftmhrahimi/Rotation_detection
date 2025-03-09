import yaml
import logging
from utils import load_images, calculate_rotation_angle

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.basicConfig(filename=config["output_log"], level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    try:
        image1_path = config["image1_path"]
        image2_path = config["image2_path"]

        image1, image2, gray1, gray2 = load_images(image1_path, image2_path)
        angle = calculate_rotation_angle(image1, image2, gray1, gray2)

        if angle is not None:
            print(f"Rotation Angle: {angle:.2f} degrees")
            logging.info(f"Rotation Angle: {angle:.2f} degrees")
        else:
            print("Unable to determine rotation angle.")
            logging.warning("Unable to determine rotation angle.")

    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
