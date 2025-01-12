# WSN-Based Heart Rate Sensor with Attack Prediction

## Overview
This project implements a **Wireless Sensor Network (WSN)** for real-time heart rate monitoring, enhanced with a **predictive model** for trend analysis and anomaly detection. The system aims to provide proactive health insights by identifying potential abnormalities in heart rate patterns, such as early signs of cardiac attacks or irregularities.

---

## Features
- **Real-time Heart Rate Monitoring**: Captures heart rate data using a WSN setup.
- **Anomaly Detection**: Uses machine learning to predict and alert for abnormal heart rate trends.
- **User Interface**: Displays data and alerts via a user-friendly mobile application.
- **Scalable System**: Designed to integrate additional sensors or parameters for comprehensive health monitoring.

---

## Technologies and Tools Used

### **Programming Languages**
- Python

### **Python Libraries**
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For implementing the machine learning model.
- **CSV**: For handling datasets during development and testing.
- **Time/Datetime**: For timestamping data.
- **PySerial** (if used): For serial communication with sensors.

### **Hardware**
- **Pulse Sensor**: (e.g., MAX30102 or SEN-11574).
- **Microcontroller**: Arduino or Raspberry Pi.
- **Wireless Communication Modules**: Bluetooth, Wi-Fi, or Zigbee.

### **Other Tools**
- **Mobile App Development Frameworks**: Flutter or React Native (if a mobile app was implemented).
- **Cloud Platforms** (optional): Google Cloud, AWS, or Firebase for data storage and analysis.

---

## System Architecture
1. **Sensor Node**: Captures heart rate data and transmits it to the microcontroller.
2. **Data Processing**: Processes raw data and cleans it for further analysis.
3. **Prediction Model**: Analyzes trends and detects anomalies using machine learning.
4. **Wireless Transmission**: Sends data to the mobile app or cloud for user access.
5. **User Interface**: Displays heart rate trends, predictions, and alerts.

---

## Installation and Usage

### **Prerequisites**
- Python 3.x installed on your system.
- Required Python libraries installed via pip:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn pyserial
  ```
- Hardware setup (sensor and microcontroller) configured and connected.

### **Steps to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/wsn-heart-rate-sensor.git
   cd wsn-heart-rate-sensor
   ```
2. Connect the sensor to the microcontroller and ensure the correct port configuration.
3. Run the Python script for data collection and prediction:
   ```bash
   python main.py
   ```
4. Launch the mobile app (if applicable) to view real-time data.
5. Monitor the results and analyze any anomalies or trends.

---

## Project Workflow
1. **Data Collection**: Captures heart rate data via the sensor.
2. **Preprocessing**: Cleans and processes the data for analysis.
3. **Training the Model**: Trains a machine learning algorithm on historical or simulated datasets.
4. **Prediction and Alerts**: Identifies anomalies and provides alerts in real-time.
5. **Visualization**: Displays data trends and predictions through graphs and charts.

---

## Results
- Achieved real-time monitoring with minimal latency.
- Predictive model accuracy: ~85% (example; replace with actual accuracy). 
- Effective detection of abnormal trends or potential health risks.

---

## Limitations
- Sensor accuracy is affected by noise and environmental factors.
- Limited computational power for handling complex models in real-time.
- Dependency on wireless network quality for seamless data transmission.

---

## Future Scope
- Integrating advanced machine learning models for improved predictions.
- Enhancing sensor precision to reduce noise and improve data reliability.
- Expanding the system to monitor additional vital parameters.
- Implementing a more interactive and intuitive mobile application.
- Scaling the system for use in clinical environments and large-scale health monitoring.

---

## Contributors
- **Shashwat Malviya** (Developer and Author)
- **Dr. Himanshu Rai Goyal** (Supervisor and Mentor)

---

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code with proper attribution.
