# NASA Space Apps Challenge 2024 [Noida]

#### Team Name - Tech Blazers 
#### Problem Statement - Development of map matching algorithm using AI-ML techniques to distinguish vehicular movement on highway and service roads.

#### Team Leader Email - srs37sahoo@gmail.com

## A Brief of the Prototype:
  This solution outlines a comprehensive system for vehicle tracking using GNSS (Global Navigation Satellite System) and OBU (Onboard Unit) data. Here’s a detailed breakdown of how the solution would work and its implementation:

1. Data Collection

GNSS Data: Vehicle GNSS data, which provides the geographical position (latitude, longitude), is collected in real-time.

OBU Data: Onboard Units (OBU) provide additional vehicle information such as speed, fuel usage, or diagnostics. This data is continuously collected from vehicles and stored in a database.


2. Data Preprocessing

GNSS Data (Noise Filtering): GNSS data can have noise or errors due to signal interruptions or multi-path reflections. A Kalman filter is used to smooth the data and estimate the vehicle's true position by filtering out noise.

Outlier Removal: Outliers in GNSS data that don't match typical movement patterns are detected using statistical methods or thresholding and removed to ensure consistency.

OBU Data Cleaning: OBU data is cleaned by removing any corrupted or missing data entries. Formatting includes converting it into a structured format for easier analysis.

Smoothing/Interpolation: Missing or inconsistent data points are filled or smoothed using interpolation techniques to provide continuous data, ensuring no gaps.


3. Data Integration

After preprocessing, the cleaned GNSS and OBU data are combined. This integration creates a cohesive dataset that contains positional, diagnostic, and performance data of the vehicle over time, making it easier to apply advanced analysis and machine learning.


4. Map Alignment

GIS Tools: Geographic Information System (GIS) tools are used to align the integrated data (vehicle positions) with a digital map. This process ensures that the vehicle's location is represented accurately in relation to the actual road network or geography. GIS also enables visualization and analysis of spatial data.


5. Machine Learning Model

Training the Model: With the integrated and aligned data, a machine learning model is trained. This model could be designed to predict future vehicle locations, estimate routes, or detect anomalies. Various algorithms, like regression models or neural networks, can be used depending on the objectives (e.g., tracking or forecasting vehicle movement).


6. Real-Time Data Processing

Collection: Real-time GNSS and OBU data are continuously gathered and fed into the system.

Processing: The trained machine learning model processes this real-time data, enabling real-time vehicle tracking, status updates, or anomaly detection (e.g., unusual driving patterns).


7. Data Visualization

GIS Visualization: The processed data is visualized using GIS-based tools that display vehicle locations on a digital map. This real-time visualization allows fleet managers or users to monitor the movement and status of vehicles directly.


8. Data Privacy & Security

Encryption: Since the system deals with sensitive vehicle data, encryption protocols are applied to protect the privacy of data during storage and transmission. This could involve encrypting the data using symmetric or asymmetric key algorithms.

Security: Secure data transmission over networks ensures that data is protected from unauthorized access. Techniques such as HTTPS, VPNs, or other secure network protocols will be utilized. Additionally, role-based access control ensures that only authorized personnel can view or manipulate the data.


Summary of Implementation:

Data Collection and Preprocessing: GNSS and OBU data are collected in real-time. Preprocessing cleans the data, removes outliers, and ensures accuracy using Kalman filtering.

Integration and Alignment: The clean data is integrated and aligned to digital maps using GIS tools.

Model Training: A machine learning model is trained on this integrated and aligned data to enable accurate tracking or prediction.

Real-Time Tracking: The model processes real-time data and provides insights (like vehicle tracking or anomaly detection) in real-time.

Visualization: Vehicle locations are displayed on a GIS map for easy monitoring.

Privacy and Security: Encryption and security measures are applied to protect the integrity and privacy of data.


This system offers a robust framework for accurate, secure, and real-time vehicle tracking and data analysis.


## Code Execution Instruction:
  *[If your solution is **not** application based, you can ignore this para]
  
 *The Repository must contain your **Execution Plan PDF**.
