#!/usr/bin/python3

# ================================================================
# DD_DOB.py
# Fractional Sliding Mode Control based on Data-Driven Disturbance Observer
# 2-DOF Motor System via Arduino
#
# Causal step indexing (matches paper exactly):
#   read y(k+1)
#   compute y(k+1), Dy = y(k+1) - y(k)
#   update A(k+1) using Dy, u and d
#   compute u(k) = - rho * Apinv(k) * (rho1 y(k) + rho2 d(k))
#
# Sliding surface matches paper exactly:
#   y(k) = (e(k) - e(k-1))/T + alpha*e(k-1)
# Sampling period T = 0.008s (Windows COM17).
# ================================================================

import serial
import time
import numpy as np
import os
import ctypes

def pinv_dmpd(A, varepsilon=0.01):
    """
    Computes the damped pseudo-inverse using the varepsilon (damping) factor.
    """
    # Get dimensions for the identity matrix
    m, n = A.shape
    
    # Formula: (A^T * A + varepsilon^2 * I)^-1 * A^T
    # We use n because A^T * A is an n x n matrix
    identity = np.eye(n)
    
    # Compute the damped inverse
    damped_matrix = A.T @ A + (varepsilon**2) * identity
    A_pinv = np.linalg.inv(damped_matrix) @ A.T
    
    return A_pinv

# ---- Windows timer resolution: set to 1ms ----------------------
ctypes.windll.winmm.timeBeginPeriod(1)

# ================================================================
# Fixed parameters
# ================================================================
T   = 0.008
PI  = np.pi
TICKS_PER_REV = 4144
T_sim         = 10.0

# ================================================================
# GL weights -- global, computed once at startup (For Later Use)
# ================================================================
Nf  = 10
BUF = Nf + 2      # ring buffer size
nu  = - 0.5
W = np.zeros(Nf + 1)
W[0] = 1.0
for _j in range(1, Nf + 1):
    W[_j] = W[_j-1] * (1.0 - (1.0 + nu) / _j)

# ================================================================
# Control parameters
# ================================================================
eta0   = T**2
delta   = T**2
A0      = 150*T
A_min   = A0
A_max   = 1.0 / T
mu      = T
rho1     = 0.30 # Theoretical deadbeat gain rho = 1.0  
rho2     = 0.30*(T**(-nu))*(-nu)
u_M     = 1.0
alpha   = 5.0
vartheta = 0.2 # DOB forgetting factor, 0.0 = no memory, 1.0 = infinite memory

# ================================================================
# Serial connection
# ================================================================
port = 'COM19'
ser  = serial.Serial(port=port, baudrate=115200, timeout=0.02)
time.sleep(1)

# ================================================================
# State initialization
# ================================================================
x1 = 0.0;  x10 = 0.0
x2 = 0.0;  x20 = 0.0

e1 = 0.0;  e1_prev = 0.0
e2 = 0.0;  e2_prev = 0.0

A          = A0 * np.eye(2)

# Ring buffer: u_hist[:,head] is most recent, u_hist[:,(head-j)%BUF] is j steps back
y_hist       = np.zeros((2, BUF))
head         = 0              # points to most recent entry
u_prev       = np.zeros(2)
y_prev_vec   = np.zeros(2)
A_prev       = A0 * np.eye(2)
dob_prev     = np.zeros(2)
k0           = 0
step         = 0

ISE  = 0.0
ITAE = 0.0
ISC  = 0.0

# System Log
f = open("data_frac_smc_dd_dob.txt", "w")
os.system("cls")
print("Experiment Starts\n")

t  = 0.0
dt = T

# ================================================================
# Main Loop
# ================================================================
while True:
    t1 = time.time()
    t += dt

    # ---- Read encoders -----------------------------------------
    ser.reset_input_buffer()    # Clear any stale data before sending read command
    ser.reset_output_buffer()   # Clear any stale data before sending read command

    sel = 5
    ser.write(bytes([sel]))
    ser.reset_output_buffer()
    try:
        x1 = int(ser.readline())
        x1 = (PI * x1 / TICKS_PER_REV) - x10
    except ValueError:
        t2 = time.time(); dt = t2 - t1
        continue

    sel = 6
    ser.write(bytes([sel]))
    ser.reset_output_buffer()
    try:
        x2 = int(ser.readline())
        x2 = (PI * x2 / TICKS_PER_REV) - x20
    except ValueError:
        t2 = time.time(); dt = t2 - t1
        continue

    # ---- Reference and error -----------------------------------
    x1d = PI * np.sin(t)
    x2d = PI * np.sin(t)
    e1  = x1 - x1d
    e2  = x2 - x2d

    # ---- Sliding variable --------------------------------------
    y1 = (e1 - e1_prev) / T + alpha * e1_prev
    y2 = (e2 - e2_prev) / T + alpha * e2_prev
    y_cur_vec = np.array([y1, y2])

    # ---- Dy ----------------------------------------------------
    Dy = y_cur_vec - y_prev_vec

    # ---- A update --------------------------------------------
    if np.linalg.norm(u_prev) > 1e-6:
        eta = eta0 / (mu  + np.linalg.norm(Dy - dob)**2
                          + np.linalg.norm(u_prev)**2)
        B_k = ((1.0 - delta * eta) * np.eye(2)
               - eta * np.outer(u_prev, u_prev))
        A = A @ B_k + eta * np.outer(Dy - dob, u_prev)

    # ---- PPD reset guards --------------------------------------
    reset_fired = False
    if (np.linalg.norm(A, 'fro') < A_min
            or np.linalg.norm(A, 'fro') > A_max):
        A = A0 * np.eye(2)
        reset_fired = True
    for j in range(2):
        if (abs(A[j, j]) < A_min
                or np.sign(A[j, j]) != np.sign(A0)):
            A[j, j] = A0
            reset_fired = True
    if reset_fired:
        k0 = step
        y_hist[:] = 0.0      # clear ring buffer contents

    # Disturbance observer update
    dob = vartheta * dob_prev + (1 - vartheta) * (Dy - A_prev @ u_prev)

    # ---- GL Differintegrator -----------------------------------
    n_lag = min(Nf, step - k0)
    gl    = np.zeros(2)
    for j in range(0, n_lag + 1):
        gl += W[j] * np.tanh(100.0*y_hist[:, (head - j) % BUF])

    head = (head + 1) % BUF
    y_hist[:, head] = y_cur_vec

    # ---- Control direction ------------------------------------
    u      = - pinv_dmpd(A,1e-3) @ (rho1 * y_cur_vec + rho2 * gl + dob)
    u_norm = np.clip(u, -u_M, u_M)

    # ---- Store for next step -----------------------------------
    u_prev       = u.copy()
    y_prev_vec   = y_cur_vec.copy()
    A_prev       = A.copy()
    dob_prev     = dob.copy()
    e1_prev      = e1
    e2_prev      = e2

    # ---- PWM conversion and send to Arduino --------------------
    u1_pwm = int(255 * u_norm[0])
    u2_pwm = int(255 * u_norm[1])

    sel = 1 if u1_pwm > 0 else 2
    ser.write(bytes([sel, abs(u1_pwm)]))
    sel = 3 if u2_pwm > 0 else 4
    ser.write(bytes([sel, abs(u2_pwm)]))

    # ---- Log ---------------------------------------------------
    f.write((
        "{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t"
        "{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4f}\t{11:.4f}\n"
    ).format(t, e1, x1, x1d, y1, u1_pwm, e2, x2, x2d, y2, u2_pwm, dt))

    # ---- Performance indices -----------------------------------
    ISE  += dt * (e1**2 + e2**2)
    ITAE += dt * t * (abs(e1) + abs(e2))
    ISC  += dt * (u_norm[0]**2 + u_norm[1]**2)

    # ---- Timing ------------------------------------------------
    # time.sleep(0.003)  # small sleep to prevent CPU overload, adjust as needed
    step += 1
    t2   = time.time()
    dt   = t2 - t1

    # ---- End condition -----------------------------------------
    if t > T_sim:
        ser.reset_input_buffer()
        ser.write(bytes([1, 0]))
        time.sleep(0.05)
        ser.reset_output_buffer()
        break

# ================================================================
# Reset and close
# ================================================================
ctypes.windll.winmm.timeEndPeriod(1)
ser.write(bytes([7]))
f.close()

print(f"\n ISE  = {ISE:.4f}")
print(f"\n ITAE = {ITAE:.4f}")
print(f"\n ISC  = {ISC:.4f}")
print("\n Experiment Ends")