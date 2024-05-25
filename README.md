# PHISHING-WEBSITE-DETECTION-USING-MACHINE-LEARNING
# Project Overview
This project aims to detect phishing websites using machine learning techniques. The goal is to build a model that identifies phishing websites based on significant URL features and develop a user interface for real-time legitimacy checking.

## Problem Statement
Current systems train models based solely on the URL and cannot accommodate all website names, requiring enhancements in accuracy. Additionally, existing systems lack a specific user interface for easy access and usage.

## Solution
We developed a machine learning model that identifies phishing websites based on significant URL features and created a web interface for users to input URLs and check their legitimacy. If a URL is identified as a phishing website, the interface will display its legitimate counterpart and store the URL for further analysis.

## Dataset
- **Name:** Phishing.csv
- **Description:** Contains a collection of URLs for 11,000+ websites, each with 30 parameters and a class label identifying it as a phishing website (1) or not (-1).

## Achievements
- Achieved 97.4% accuracy with individual machine learning models.
- Improved accuracy to 97.9% using ensemble techniques (stacking random forest and gradient boosting).
- Developed a mapping dataset that relates phishing URLs to legitimate URLs.

## Tools and Technologies
- **Programming Languages:** Python
- **Libraries and Frameworks:** Scikit-learn, Pandas, NumPy
- **Development Environment:** Jupyter Notebook, PyCharm, VSCode
- **Machine Learning Techniques:** Random Forest, Gradient Boosting, Ensemble Techniques (Stacking)
- **Web Development:** Flask (or Django), HTML, CSS, JavaScript
- **Database:** SQLite (or MySQL)

## Features
- **Phishing Detection:** Classifies URLs as legitimate or phishing based on significant URL features.
- **User Interface:** Allows users to input URLs and check their legitimacy in real-time.
- **URL Mapping:** Displays legitimate URLs for identified phishing websites.
- **Data Storage:** Stores user-submitted URLs for analysis and continuous improvement of the detection system.

## Usage
**Installation:**
1. Clone the repository.
2. Install the required Python libraries using `pip install -r requirements.txt`.

**Running the Application:**
1. Start the Flask server by running `python app.py`.
2. Open your web browser and navigate to `http://localhost:5000` to access the user interface.

**Input a URL:**
1. Enter a URL in the input field and submit.
2. The application will display whether the URL is legitimate or a phishing website.
3. If the URL is identified as phishing, it will also display the corresponding legitimate URL.

## Future Work
- Enhance the model's accuracy by incorporating more sophisticated features and techniques.
- Expand the mapping dataset to cover a wider range of phishing URLs.
- Improve the user interface for better usability and user experience.

## Contributors
- [Navanisha S] -Developer
- - [Pavan L] -Developer
  - - [Pooja G R] -Developer
    - - [Pooja K S] -Developer

## License
This project is licensed under the MIT License - see the LICENSE file for details.

