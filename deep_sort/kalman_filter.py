import numpy as np
import scipy
class Kalman_Filter:
    def __init__(self):
        ndim, dt = 4, 1

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim+i] = dt

        self._update_mat = np.eye(ndim, 2*ndim)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    #predict n + 1 position
    def init(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)

        mean = np.r_[mean_pos,mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]

        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        
        motion_cov = np.diag(np.square(np.r_[std_pos,std_vel]))

        mean = np.dot(mean, self._motion_mat)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
    def project(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        
        mean = np.dot(self._update_mat, mean)
        innovation_cov = np.diag(np.square(std_pos))

        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance
    
    def update(self, measurement, mean, covariance):
        projected_mean, projected_cov, = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        
        innovative = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain,innovative)

        new_covariance = np.dot((np.ones_like(kalman_gain) - kalman_gain),projected_cov)

        return new_mean, new_covariance