# Stewart Platform Ball Balancer

A real-time control system that uses computer vision to balance a ball on a 3-DOF Stewart Platform. 

<a href="https://youtube.com/shorts/gjLFBipaO6I?feature=share">
   <img alt="View " title="See it in Action!" src="https://custom-icon-badges.demolab.com/badge/-Video Demo-blue?style=for-the-badge&logo=browser&logoColor=white"/></a> 

<p>
<img width="650" height="600" alt="image" src="https://github.com/user-attachments/assets/3d104806-19c8-4d84-9484-973e87339cb6" />
</p>

## Tech Stack
* **Languages:** Python, C++
* **Computer Vision:** OpenCV
* **Hardware:** Arduino/Servos via PySerial
* **Interface:** Tkinter & Matplotlib

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/ae1a0516-99d0-44c6-8fbe-adc3a3ad6db7" />


## Key Features
* **Automated Calibration:** Interactive tool to map ball color (HSV), platform center, and motor orientations using <a href="https://docs.wpilib.org/en/stable/docs/software/vision-processing/apriltag/apriltag-intro.html">AprilTags</a>
* **Real-Time Tracking:** HSV-based detection that converts pixel coordinates into physical meters.
* **PID Control:** A GUI-driven controller for live-tuning $K_p$, $K_i$, and $K_d$ gains.

## Quick Start
1. **Install Dependencies:**
   ```bash
   pip install opencv-python numpy pyserial matplotlib
