# 🚢 Titanic Survival Predictor

An interactive Machine Learning web application built using Python and Streamlit to predict the survival probability of Titanic passengers.

## 📌 Project Overview

This project uses a Random Forest Classifier trained on historical Titanic passenger data. Users can enter passenger details and receive an estimated survival prediction through an interactive web interface.

## ✨ Features

- Interactive Streamlit web interface
- Passenger ticket class selection
- Gender selection
- Age and ticket fare sliders
- Real-time survival prediction
- Estimated survival probability
- Machine Learning model loaded using Joblib

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Random Forest Classifier

## 📂 Project Structure

```text
Project-05-Titanic-Survival-Predictor/
│
├── app.py
├── train_model.py
├── model.joblib
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Parthgodage/Project-05-Titanic-Survival-Predictor.git
```

2. Open the project folder.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## 🤖 Machine Learning Model

Algorithm: Random Forest Classifier

Dataset: Titanic passenger dataset

Input Features:
- Passenger Class
- Gender
- Age
- Ticket Fare

## 👨‍💻 Developer

**Parth Godage**

Data Science Student

GitHub: https://github.com/Parthgodage

## ⚠️ Disclaimer

This project is developed for educational purposes. Predictions are estimates based on historical data and are not guarantees of actual outcomes.
