import numpy as np
import math as m

def Rx(theta):
    theta = m.radians(theta)
    return np.array([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])

def Ry(theta):
    theta = m.radians(theta)
    return np.array([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
    theta = m.radians(theta)
    return np.array([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])

def Rx_rad(theta):
    return np.array([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])

def Ry_rad(theta):
    return np.array([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])

def Rz_rad(theta):
    return np.array([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])

def part1():
    r_BL = np.array([[2.57], [-0.52], [1.32]])
    C_BL = (Rz(-90) @ Ry(-23) @ Rx(-10))
    Cr_BL = np.hstack((C_BL, r_BL))
    T_BL = np.vstack((Cr_BL, np.array([0, 0, 0, 1])))
    print(f"T_BL: \n{T_BL}")
    print(f"T_BL inv: \n{np.linalg.inv(T_BL)}")

    p_L = np.array([[3.64], [8.30], [2.45]])
    p_L_hom = np.vstack((p_L, np.array([1])))
    p_B_hom = T_BL @ p_L_hom
    print(f"p_B_hom \n{p_B_hom}")

def part2():
    k1 = -0.369
    k2 = 0.197
    k3 = 1.35e-3
    t1 = 5.68e-4
    t2 = -0.068
    fx = 959.79
    fy = 956.93
    px = 696.02
    py = 224.18

    def plumb_bob(pt):
        x = pt[0][0]
        y = pt[1][0]
        r = m.sqrt(x*x + y*y)
        rad_coef = 1 + k1*r**2 + k2*r**4 + k3*r**6
        rad_dist = np.array([[rad_coef, rad_coef, 1]]).T
        tan_dist = np.array([[2*t1*x*y + t2*(r**2 + 2*x**2), 2*t2*x*y + t1*(r**2 + 2*y**2), 0]]).T
        return np.multiply(rad_dist, pt) + tan_dist

    C_CB = Ry(-90)
    r_CB = np.array([[1.06], [-0.11], [-2.82]])
    Cr_CB = np.hstack((C_CB, r_CB))
    T_CB = np.vstack((Cr_CB, np.array([0, 0 ,0 ,1])))
    print(f"T_CB: \n{T_CB}")

    K = np.array([[fx, 0, px],
                 [0, fy, py],
                 [0, 0, 1]])

    p_B_hom = np.array([[4.47, -0.206, 0.731, 1]]).T
    p_C_hom = T_CB @ p_B_hom
    p_C = p_C_hom[0:3, :]
    p_C_norm = p_C/p_C[2]
    p_C_distort = plumb_bob(p_C_norm)
    p_C_pix = K @ p_C_distort
    print(f"p_C_pix: \n{p_C_pix}")

    pix_loc = np.round(p_C_pix[0:2, :])
    print(f"pix_loc: \n{pix_loc}")



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    part1()
    part2()
