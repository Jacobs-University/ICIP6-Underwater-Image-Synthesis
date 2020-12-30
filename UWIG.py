# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 21:52:02 2020

@author: abhie
"""

import numpy as np

#load data
at_coef = np.genfromtxt('water_attenuation_coef.csv', delimiter=',')
Sc = np.genfromtxt('spectral_response_cams.csv', delimiter=',')
E0 = np.genfromtxt('irradiance.csv', delimiter=',')

def calc_beta_vert(c,d,water,camera): #c = [0,1,2] for red, green, blue respectively, d = vertical distance in meter
    numerator = np.sum(Sc[:,camera*3+c] * E0)
    denominator = np.sum(Sc[:,camera*3+c] * E0 * np.exp(-at_coef[:,water]*d))
    beta = np.log(numerator/denominator)/d
    
    return beta

def calc_beta_horz(c,d,D,water,camera): #D = horizontal distance in meter, c and d same as above
    numerator = np.sum(Sc[:,camera*3+c] * E0 * np.exp(-at_coef[:,water]*d))
    denominator = np.sum(Sc[:,camera*3+c] * E0 * np.exp(-at_coef[:,water]*(d+D)))
    betah = np.log(numerator/denominator)/D
    
    return betah

def calc_transmisssion_map(beta,d):
    return np.exp(-beta*d)

#water type = 0,1,,,8,9 [corresponding to Jerlovs water types: I,IA,IB,II,III,1C,3C,7C,9C]
#camera type = 0,1,2 [0 : Nikon, 1 : Canon, 2 : Mobile(grasshopper)]
#d = vertical distace from surface of water
#image matrix in BGR format (as read in by cv2)
def generate_uw(image, depth_map, water_type, camera_type, d):
    
    #initialization - vary between [0.7,0.7,0.4] to [0.8,0.8,0.5]
    B_bgr = [0.75,0.75,0.45]
    
    #c = 0 = Blue channel; c = 1 = Green channel; c = 2 = Red channel
    #calculating formulas
    beta_vert = []
    U_img = np.zeros_like(image)

    for c in range(0,3):

        beta_vert.append(calc_beta_vert(2-c, d, water_type, camera_type))
        trans_map_vert = calc_transmisssion_map(beta_vert[c], d)


        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                beta_horz = calc_beta_horz(2-c, d, depth_map[i,j], water_type, camera_type)
                trans_map_horz = calc_transmisssion_map(beta_horz, depth_map[i,j])

                U_img[i,j,c] = image[i,j,c]*trans_map_vert*trans_map_horz + B_bgr[c]*trans_map_vert*(1-trans_map_horz)
    
    return U_img