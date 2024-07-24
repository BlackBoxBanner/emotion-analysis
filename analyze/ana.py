import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Read the CSV file
df0df = pd.read_csv("../result/deepface_analysis_results_0.csv")

# Iterate over each row in the DataFrame
for index, row in df0df.iterrows():
    # Read the image
    image = cv2.imread(f"../{row['file_path']}")

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image_rgb)
    plt.title(f"Image: {row['file_path']}, Face: {row['face_number']}")
    plt.axis('off')  # Hide axis
    plt.show()