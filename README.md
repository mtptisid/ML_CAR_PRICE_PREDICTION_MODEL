# Car Price Predictor

## Overview

The **Car Price Predictor** is a web application that predicts the price of a car based on user-provided details. This application uses a machine learning model to estimate car prices, and it provides a user-friendly interface to input data and receive predictions.

## Features

- **Predict Car Prices**: Estimate the price of a car based on features such as company, model, year of purchase, fuel type, and kilometers driven.
- **Dynamic Model Loading**: Car models are dynamically loaded based on the selected car company.
- **Responsive Design**: The application is designed to be responsive and user-friendly across different devices.

## Technologies Used

- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Backend**: Flask
- **Machine Learning**: Python with scikit-learn
- **Dependencies**: TensorFlow.js for future integrations (optional)

## Installation

To set up and run the Car Price Predictor locally, follow these steps:

### Prerequisites

- Python 3.x
- Flask
- scikit-learn
- TensorFlow (if used in the future)

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/mtptisid/ML_CAR_PRICE_PREDICTION_MODEL.git
    cd car-price-predictor
    ```

2. **Set Up a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare Your Model**

    Ensure your machine learning model (`LinearRegressionModel.pkl`) and the dataset (`Cleaned_Car_data.csv`) are in the appropriate directories.

5. **Run the Application**

    ```bash
    python application.py
    ```

6. **Open Your Browser**

    Navigate to `http://127.0.0.1:5000` to view and interact with the application.

## Usage

1. **Select Company**: Choose the car company from the dropdown menu.
2. **Select Model**: Select the car model based on the chosen company.
3. **Enter Details**: Provide the year of purchase, fuel type, and kilometers driven.
4. **Predict Price**: Click the "Predict Price" button to get the estimated price of the car.

## Development

To contribute to the development of this project:

1. **Fork the Repository**
2. **Create a New Branch**

    ```bash
    git checkout -b feature/your-feature
    ```

3. **Make Changes**
4. **Commit Your Changes**

    ```bash
    git add .
    git commit -m "Add your commit message"
    ```

5. **Push to Your Branch**

    ```bash
    git push origin feature/your-feature
    ```

6. **Create a Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or feedback, please contact:

- **Email**: msidrm455@gmail.com
- **GitHub**: [mtptisid](https://github.com/mtptisid)
