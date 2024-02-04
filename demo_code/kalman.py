import numpy as np

class KalmanFilter:
    def __init__(self):
        # State transition matrix
        self.A = np.array([[1, 0], [0, 1]])

        # Observation matrix
        self.H = np.array([[1, 0], [0, 1]])

        # State covariance matrix
        self.P = np.eye(2)

        # Process noise covariance matrix
        self.Q = np.eye(2) * 1e-5

        # Measurement noise covariance matrix
        self.R = np.eye(2) * 1e-3

    def predict(self, x):
        x_pred = np.dot(self.A, x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return x_pred

    def update(self, x_pred, z):
        y = z - np.dot(self.H, x_pred)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        x_est = x_pred + np.dot(K, y)
        self.P = np.dot((np.eye(2) - np.dot(K, self.H)), self.P)
        return x_est
        
# Example usage
if __name__ == "__main__":
    kf = KalmanFilter()

    # Initial state
    x = np.array([0, 0])

    # Example measurements
    measurements = [
        (10, 10),
        (20, 20),
        (30, 30),
        (40, 40),
        (50, 50),
    ]

    for z in measurements:
        x_pred = kf.predict(x)
        x = kf.update(x_pred, z)
        print(f"Estimated coordinates: {x}")