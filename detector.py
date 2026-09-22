import os
from pathlib import Path
from collections import Counter

import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


# --------------------------------------------------
# Connect to the pretrained Roboflow model
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY was not found in .env")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
).configure(
    InferenceConfiguration(
        confidence_threshold=0.25,
        iou_threshold=0.45,
        api_key_transport="header"
    )
)

model_id = "recyclable-waste-tgt8a/1"


# --------------------------------------------------
# Load one test image
# --------------------------------------------------

image_path = Path(
    "test/images/"
    "IMG_6404_JPG.rf.61f64fcedbb97584772c98d8b4ff573f.jpg"
)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read {image_path}")

image_height, image_width = image.shape[:2]


# --------------------------------------------------
# Run the pretrained detector
# --------------------------------------------------

result = client.infer(
    str(image_path),
    model_id=model_id
)

predictions = result.get("predictions", [])

print(f"Detections found: {len(predictions)}")


# --------------------------------------------------
# Count detections by class
# --------------------------------------------------

class_counts = Counter(
    prediction["class"].strip().lower()
    for prediction in predictions
)

recyclable_count = class_counts["recyclable"]
non_recyclable_count = class_counts["non-recyclable"]

print()
print(f"Image: {image_path.name}")
print()
print(f"Recyclable: {recyclable_count}")
print(f"Non-Recyclable: {non_recyclable_count}")


# --------------------------------------------------
# Draw every returned bounding box
# --------------------------------------------------

for prediction in predictions:
    center_x = prediction["x"]
    center_y = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]

    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    # Keep coordinates inside the image.
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    class_name = prediction["class"]
    normalized_class = class_name.strip().lower()
    confidence = prediction["confidence"]

    if normalized_class == "recyclable":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    label = f"{class_name}: {confidence:.2f}"

    # Draw the bounding box.
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    # Measure the label so a background can be drawn.
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    text_y = max(y1 - 8, text_height + 8)

    # Draw a filled background behind the label.
    cv2.rectangle(
        image,
        (x1, text_y - text_height - 8),
        (min(x1 + text_width + 8, image_width - 1), text_y + 4),
        color,
        -1
    )

    cv2.putText(
        image,
        label,
        (x1 + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    print(label)


# --------------------------------------------------
# Display detection counts on the image
# --------------------------------------------------

cv2.rectangle(
    image,
    (5, 5),
    (310, 75),
    (0, 0, 0),
    -1
)

cv2.putText(
    image,
    f"Recyclable: {recyclable_count}",
    (15, 32),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    f"Non-Recyclable: {non_recyclable_count}",
    (15, 62),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)


# --------------------------------------------------
# Save the completed image
# --------------------------------------------------

Path("results").mkdir(exist_ok=True)

output_path = Path("results/test_result.jpg")

success = cv2.imwrite(str(output_path), image)

if not success:
    raise RuntimeError(f"Could not save {output_path}")

print()
print(f"Saved result to {output_path}")