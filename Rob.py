#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install openpyxl


# In[2]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import math
from scipy.optimize import newton
import warnings
warnings.simplefilter("ignore")

df1=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsb.xlsx", sheet_name='Sheet3')
df2=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsb.xlsx", sheet_name='New')
df3=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsb.xlsx", sheet_name='Cooling water')
df4=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsb.xlsx", sheet_name='Trial and Error method')

df1['Date'] = df1['Date'].astype(str)

header = st.container()
# header = st.beta_container()
# header = st.beta_container()
with header:
    st.title("Heat and mass transfer processes in a condensing heat exchanger")

RE = st.selectbox("Enter a value for Reynolds number:",(3000,5000,10000))


MF = st.selectbox("Enter a value for the mass fraction of water vapour:",(10,20,30))

T = st.selectbox("Enter a value for the inlet temperature of humid air:",(85,145,205))

RF = st.selectbox("Enter a value for ratio of flow rates between cold and hot fluids :",(1,2,3,4))

def parameters(RE, MF, T, RF):
    return df1.query("`Re` == @RE & `Mass Fraction` == @MF & `Mixture tin, oC` == @T & `Ratio of water/mixture flow rates` == @RF")

result = parameters(RE, MF, T, RF)
if result.shape[0]>1:
    st.write("The number of system with the chosen condition is ",np.round(result.shape[0],1))
    e = st.slider(
        "Choose system number",
        0.0, np.round(result.shape[0],1),0)
    st.write("Values:", e)
elif result.shape[0]==1:
    e=result.shape[0]
elif result.shape[0]==0:
    st.write("There is no system with these parameters")
    st.stop()


# In[3]:


Humid_air = df1.loc[e ,['Temperature decrease of the mixture_1',
 'Temperature decrease of the mixture_2',
 'Temperature decrease of the mixture_3',
 'Temperature decrease of the mixture_4',
 'Temperature decrease of the mixture_5',
 'Temperature decrease of the mixture_6',
 'Temperature decrease of the mixture_7',
 'Temperature decrease of the mixture_8']]

Cooling_water = df1.loc[e ,['Temperature increase of the cooling water_1',
 'Temperature increase of the cooling water_2',
 'Temperature increase of the cooling water_3',
 'Temperature increase of the cooling water_4',
 'Temperature increase of the cooling water_5',
 'Temperature increase of the cooling water_6',
 'Temperature increase of the cooling water_7',
 'Temperature increase of the cooling water_8']]

Wall_temp = df1.loc[e ,['Tube_coil_surface_temp1',
 'Tube_coil_surface_temp2',
 'Tube_coil_surface_temp3',
 'Tube_coil_surface_temp4',
 'Tube_coil_surface_temp5',
 'Tube_coil_surface_temp6',
 'Tube_coil_surface_temp7',
 'Tube_coil_surface_temp8']]

dew_point = df1.loc[e,['Dew point_1','Dew point_2','Dew point_3','Dew point_4','Dew point_5','Dew point_6',
'Dew point_7','Dew point_8']]


# In[4]:


e = result.index[0]

Cooling_water = df1.loc[e ,['Temperature increase of the cooling water_1',
 'Temperature increase of the cooling water_2',
 'Temperature increase of the cooling water_3',
 'Temperature increase of the cooling water_4',
 'Temperature increase of the cooling water_5',
 'Temperature increase of the cooling water_6',
 'Temperature increase of the cooling water_7',
 'Temperature increase of the cooling water_8',
'Temperature increase of the cooling water_9']].values
Flue_gas = df1.loc[e ,['Temperature decrease of the mixture_1',
 'Temperature decrease of the mixture_2',
 'Temperature decrease of the mixture_3',
 'Temperature decrease of the mixture_4',
 'Temperature decrease of the mixture_5',
 'Temperature decrease of the mixture_6',
 'Temperature decrease of the mixture_7',
 'Temperature decrease of the mixture_8',
'Temperature decrease of the mixture_9']].values


m_cd = 0
y_H2o= []
Sat_temp = []
Outlet_temp_c = []
Operating_tempc = []
Water_density = []
Mass_flowrate = []
Water_velocity = []
Water_Dynamic_viscosity = []
Water_Reynolds = []
Water_thermal_conductivity = []
Water_Nusselt_number = []
Water_heat_Transfer_coefficient = []
Water_specific_heat = []
# Wall_temperature1 =[]
Wall_temperature1 = df1.loc[e ,['Tube_coil_surface_temp1',
 'Tube_coil_surface_temp2',
 'Tube_coil_surface_temp3',
 'Tube_coil_surface_temp4',
 'Tube_coil_surface_temp5',
 'Tube_coil_surface_temp6',
 'Tube_coil_surface_temp7',
 'Tube_coil_surface_temp8']].values
Wall_temperature2 = []
Op_temp_air = []
Mass_of_diffusivity = []
Density_air = []
FlowRate_air = []
Velocity_air = []
Specific_heat_air = []
Reynolds_air = []
Thermal_conductivity_air = []
Thermal_diffusivity_air = []
Nusselt_air = []
Heat_transfer_air = []
Latent_heat_air = []
Lewis_air = []
Temperature_interface = []
Vapour_mole_interface = []
Mass_transfer_coefficient_air = []
Logarithmic_mole_average = []
Inlet_temp_air = []
Outlet_temp_air = []
Inlet_temp_water = []
Mass_fraction = []
Condensation_rate = []
numbering = []
m_frac = df1.loc[e,'Mass Fraction']/100
steam_flowrate = df1.loc[e,'Vapour flow rate, kg/h']/3600 
air_flowrate = df1.loc[e,'Ratio of water/mixture flow rates']
M_frac = m_frac

for _ in range(8):
    #Calculate y_h2o:
    print("# of iteration: ",_+1)
    m_frac = M_frac
    M_h2o = 18.015
    M_g = 28.96
    y_h2o = (float(m_frac) / M_h2o) / ((float(m_frac) / M_h2o) + ((1 - float(m_frac)) / M_g))
    y_H2o.append(y_h2o)
    print("Mole fraction of water vapor:",y_h2o)

    P_w = y_h2o * 1 
    print("The partial pressure of water vapor (P_w):",P_w,"atm")

    pressure = [0.15,0.2,0.25,0.5]
    temperature = [53.983,60.073,64.980,81.339]



    T_sat = np.interp(P_w, pressure, temperature)
    Sat_temp.append(T_sat)
    print("Partial pressure of water vapour:", P_w,'atm')
    print("Partial pressure of air:", 1-P_w,'atm')
    print("The saturation temperature of water vapour:",T_sat)


    T_c = (Cooling_water[_] + Cooling_water[_+1]) / 2
    T_c = float(T_c)
    print("The operational temperature of cooling water is ",T_c,"°C")
    T_cout = Cooling_water[_]
    print("The outlet temperature of cooling water is ",T_cout,"°C")

    T_cout = float(T_cout)
    T_cin = Cooling_water[_+1]
    print("The intlet temperature of cooling water is ",T_cin,"°C")
    Outlet_temp_c.append(T_cout)
    Operating_tempc.append(T_c)


    # Calculating the density of the cooling water
    values1 = [0.1, 1, 4, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
               85, 90, 95, 100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
               320, 340, 360, 373.946]
    values2 = [999.85, 999.9, 999.97, 999.7, 999.1, 998.21, 997.05, 995.65, 994.03, 992.22,
               990.21, 988.04, 985.69, 983.2, 980.55, 977.76, 974.84, 971.79, 968.61, 965.31,
               961.89, 958.35, 950.95, 943.11, 926.13, 907.45, 887, 864.66, 840.22, 813.37, 783.63,
               750.28, 712.14, 667.09, 610.67, 527.59, 322]

    rho_c = np.interp(T_c, values1, values2)
    Water_density.append(rho_c)
    print("Density of cooling water {} kg/m³".format(rho_c))
 
    m_c = df1.loc[e,'Cooling water flow rate, l/h'] * rho_c/(3600*1000*3) 
    Mass_flowrate.append(m_c)
    A_c = ((0.014**2)*math.pi)/4 
    v_c = m_c / (rho_c*A_c) 
    Water_velocity.append(v_c)
    print("The velocity on the cooling water side {} m/s".format(np.round(v_c,4)))
    print("Mass flow rate on the cooling water side {} Kg/s".format(np.round(m_c,4)))
    print("Area of the cooling water side {} m²".format(np.round(A_c,4)))

    T = T_c +273 
    u_c = 0.0000000000330128*(T**(4)) - 0.0000000448315*(T**(3)) + 0.0000228887*(T**(2)) - 0.00521331*(T) + 0.447905 #Dynamic viscosity
    Water_Dynamic_viscosity.append(u_c)
    print("Dynamic viscosity of the cooling water side {} μPa.s".format(np.round(u_c,5)))
    D_i = 0.014 #m
    Re_c = (rho_c * v_c * D_i) / u_c    
    Water_Reynolds.append(Re_c)
    print("Reynolds number for the cooling water side {}".format(np.round(Re_c,4)))

    my_list1 = [0.01, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.6]
    my_list2 = [555.75, 578.64, 598.03, 614.5, 628.56, 640.6, 650.91, 659.69, 667.02, 672.88, 677.03]
    point = float(T_c)
    k_c = np.interp(point, my_list1, my_list2)/1000 # W/m·K, thermal conductivity
    Water_thermal_conductivity.append(k_c)
    print("Thermal conductivity of the cooling water {} W/m.K".format(np.round(k_c,4)))

    values1 = [0.01, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    values2 = [4.2199, 4.1955, 4.1844, 4.1816, 4.1801, 4.1796, 4.1815, 4.1851, 4.1902, 4.1969, 4.2053, 4.2157]
    point = float(T_c)
    c_pc = np.interp(point, values1, values2)*1000 
    Water_specific_heat.append(c_pc)
    print('Specific heat of the cooling water side {} j/Kg.K'.format(np.round(c_pc,4)))

    alpha_c = k_c / (rho_c*c_pc) 
    pr_c =  (u_c/rho_c) /  (alpha_c)
    print("Prandlt number for the cooling water side:", np.round(pr_c,4))
    if Re_c < 3000:
        print('The flow is Laminar')   
        x = 0.30 
        x_ = ((2*x)/D_i) / (Re_c * pr_c) 
        print("Non-dimensional entry length x_:", x_)
        M_Nu = [19.29,12.09,8.92,5.81,4.86,4.64,4.15]
        L_Nu = [12.8,8.03,6,4.17,3.77,3.71,3.66]
        x = [0.001,0.004,0.01,0.04,0.08,0.1,0.2]
        x_inter = x_ 
        Mean_Nu = np.interp(x_inter, x, M_Nu)
        Local_Nu = np.interp(x_inter, x, L_Nu)
        print("Local Nusselt number:", Mean_Nu)
        print("Mean Nusselt number:", Local_Nu)
#         Nu_c = input("The value for Nusselt number:")
        Water_Nusselt_number.append(Mean_Nu)
        h_c= (float(Mean_Nu)*k_c) / D_i
        Water_heat_Transfer_coefficient.append(h_c)
        print("The heat transfer coefficient of the cooling water {} W/m\u00B2K".format(np.round(h_c,4))) 
    else:
        print("The flow is Turbulent")
        if (Re_c <= 2*(10**4)):
            f = 0.316 * Re_c **-0.25
            print("Friction factor1:", f)
            Nu_c = (f/8) * (Re_c - 1000) * pr_c / (1 + 12.7 * (f/8)**0.5 * (pr_c**(2/3) - 1))
            Water_Nusselt_number.append(Nu_c)
            print("Nusselt number for the cooling water side:", Nu_c)
            h_c= (float(Nu_c)*k_c) / D_i
            Water_heat_Transfer_coefficient.append(h_c)
            print("The heat transfer coefficient of the cooling water {} W/m\u00B2K".format(np.round(h_c,4))) 
        else:
            f = (0.790 * math.log(Re_c) - 1.64)**-2
            print("Friction factor2:", f)
            Nu_c = (f/8) * (Re_c - 1000) * pr_c / (1 + 12.7 * (f/8)**0.5 * (pr_c**(2/3) - 1))
            Water_Nusselt_number.append(Nu_c)
            print("Nusselt number for the cooling water side:", Nu_c)
            h_c= (float(Nu_c)*k_c) / D_i
            Water_heat_Transfer_coefficient.append(h_c)
            print("The heat transfer coefficient of the cooling water {} W/m\u00B2K".format(np.round(h_c,4))) 

    #calculation of the wall temperature
    delta_Ai = 0.364*math.pi*D_i
    numerator = m_c*c_pc*(T_cout - T_cin)*3
    Denominator = h_c*delta_Ai*3
    T_w = Wall_temperature1[_]
    print("The wall temperature:", np.round(T_w,4),'°C')
    if T_w<T_sat:
        print("There will be condensation")
    else:
        print("There is no condensation")

    T_g = (Flue_gas[_] + Flue_gas[_+1]) / 2
    T_g = float(T_g)
    print("The operational temperature of the flue gas is ",T_g,"°C")
    Op_temp_air.append(T_g)

    p =  101325 
    R_air =8.314  
    R_water = 8.314 
    T = float(T_g)+273
    rho_air = ((p*M_g/1000)/(R_air*T))*(1-y_h2o)
    rho_water = ((p*M_h2o/1000)/(R_water*T))*y_h2o
    print("Partial density of air {} kg/m\u00B3".format(rho_air))
    print("Partial density of water vapour {} kg/m\u00B3".format(rho_water))
    y_air = 1-y_h2o
    rho_g = rho_water + rho_air
    Density_air.append(rho_g)
    print("Density of the air side {} kg/m\u00B3".format(rho_g))

    m_g = (df1.loc[e,'Mixture  (air+vapour) flow rate, kg/h']/60**2) - m_cd  
    FlowRate_air.append(m_g)
    D_o = 0.018 
    A_gap = 0.011 
    v_g = m_g / (rho_g*A_gap)
    Velocity_air.append(v_g)
    print("The velocity of the air side {} m/s".format(v_g))

    values1 = [0, 6.9, 15.6, 26.9, 46.9, 66.9, 86.9, 107, 127, 227, 327, 427, 527, 627]
    values2 = [1.006, 1.006, 1.006, 1.006, 1.007, 1.009, 1.01, 1.012, 1.014, 1.03, 1.051, 1.075, 1.099, 1.121]
    point = float(T_g)
    cp_Air = np.interp(point, values1, values2)
    values3 = [-23, 2, 27, 52, 77, 102, 127, 177, 227, 277, 327, 377, 427, 477, 527, 577, 627, 677, 727]
    values4 = [1.855, 1.859, 1.864, 1.871, 1.88, 1.89, 1.901, 1.926, 1.954, 1.984, 2.015, 2.047, 2.08, 2.113, 2.147, 2.182, 2.217, 2.252, 2.288]
    point = float(T_g)
    cp_water = np.interp(point, values3, values4)
    M_m = y_air*M_g + y_h2o*M_h2o
    c_pg = ((M_g/M_m)*y_air*cp_Air + (M_h2o/M_m)*y_h2o*cp_water)
    print ("Specific heat of the air side {} Kj/Kg.K".format(np.round(c_pg,4)))
    Specific_heat_air.append(c_pg)

    values1 = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200, 225, 300]
    values2 = [0.00001715, 0.0000174, 0.00001764, 0.00001789, 0.00001813, 0.00001837, 
                     0.0000186, 0.00001907, 0.00001953, 0.00001999, 0.00002088, 0.00002174, 
                     0.00002279, 0.0000238, 0.00002478, 0.00002573, 0.00002666, 0.00002928]
    u_air = np.interp(point, values1, values2) 
    values3 = [ 17.51, 24.1, 28.98, 32.9, 36.18, 39.02, 41.53, 43.79, 45.83, 60.09, 69.13, 75.89, 81.35, 85.95, 89.96,
                   93.51, 96.71, 99.63, 102.32, 104.81, 107.13, 109.32, 111.37, 111.37, 113.32, 115.17, 116.93,
                   118.62, 120.23, 123.27, 126.09, 128.73, 131.2, 133.54, 138.87, 143.63, 147.92, 151.85,
                   155.47, 158.84, 161.99, 164.96, 167.76, 170.42, 172.94, 175.36, 177.67, 179.88, 184.06,
                   187.96, 191.6, 195.04, 198.28, 201.37, 204.3, 207.11, 209.79, 212.37, 214.85, 217.24,
                   219.55, 221.78, 223.94, 226.03, 228.06, 230.04, 231.96, 233.84]
    values4 = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.000011, 0.000011,
                        0.000011, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000013,
                        0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013,
                        0.000013, 0.000013, 0.000013, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014,
                        0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015,
                        0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016,
                        0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000017]
    u_water = np.interp(point, values3, values4) 
    Q_av = (math.sqrt(2) / 4) * (1 + (M_g / M_h2o)) ** -0.5 * ((1 + math.sqrt(u_air / u_water)) * (M_h2o / M_g) ** 0.25) ** 2
    Q_va = (math.sqrt(2) / 4 )*( 1 + (M_h2o / M_g)) ** -0.5 * ((1 + math.sqrt(u_water / u_air)) * (M_g / M_h2o) ** 0.25) ** 2


    u_g = ((y_air*u_air)/((y_air) + (y_h2o*Q_av))) + ((y_h2o*u_water)/((y_h2o) + (y_air*Q_va)))
    print("Coefficient 1av {}".format(Q_av))
    print("Coefficient 2va {}".format(Q_va))
    print("Dynamic viscosity of dry air {} μPa s".format(np.round(u_air,10)))
    print("Dynamic viscosity of vapour {} μPa s".format(np.round(u_water,10)))
    print("Dynamic viscosity of the mixture {} μPa s".format(np.round(u_g,10)))

    Re_g = (rho_g*v_g*D_o)/u_g
    Reynolds_air.append(Re_g)
    print("Reynolds number on the air side:",np.round(Re_g,4))


    T = float(T_g)
    values = [-190, -150, -100, -75, -50, -25, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150,
              175, 200, 225, 300, 412, 500, 600, 700, 800, 900, 1000, 1100]
    numbers = [7.82, 11.69, 16.2, 18.34, 20.41, 22.41, 23.2, 23.59, 23.97, 24.36, 24.74, 25.12, 25.5, 25.87, 26.24,
               26.62, 27.35, 28.08, 28.8, 30.23, 31.62, 33.33, 35, 36.64, 38.25, 39.83, 44.41, 50.92, 55.79, 61.14,
               66.32, 71.35, 76.26, 81.08, 85.83]
    k_g_air = np.interp(T,values,numbers)/1000
    numbers1 = [0.01, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 320, 340, 360]
    numbers2 = [0.0171, 0.0173, 0.0176, 0.0179, 0.0182, 0.0186, 0.0189, 0.0192, 0.0196, 0.02, 0.0204, 0.0208, 0.0212, 0.0216, 0.0221, 0.0225, 0.023, 0.0235, 0.024, 0.0246, 0.0251, 0.0262, 0.0275, 0.0288, 0.0301, 0.0316, 0.0331, 0.0347, 0.0364, 0.0382, 0.0401, 0.0442, 0.0487, 0.054, 0.0605, 0.0695, 0.0836, 0.11, 0.178]
    k_g_vapour = np.interp(T,numbers1,numbers2)
    k_g = ((1-y_h2o)*k_g_air)/((1-y_h2o)+Q_av*y_h2o) + (y_h2o*k_g_vapour)/(y_h2o+(1-y_h2o)*Q_va)  # (W/m k) thermal conductivity of air
    Thermal_conductivity_air.append(k_g)
    print("Thermal conductivity of air {} W/m k".format(np.round(k_g,4)))
    alpha_g = k_g / (rho_g*c_pg*1000) # (m2/s) thermal diffusivity
    Thermal_diffusivity_air.append(alpha_g)
    print("Thermal diffusivity {} m\u00B2/s".format(np.round(alpha_g,8)))
                               
    pr_g =  (u_g/rho_g) /  (alpha_g)
    print("Prandlt number for the air side:", np.round(pr_g,4))

    if (Re_g <= 2*10**6) and (Re_g >= 1000) and (0.7 <= np.round(pr_g,1)) and (np.round(pr_g,1) <= 500):
        c = 0.27 # 
        m = 0.63 # 
        Nu_g = c*(Re_g**m)*(pr_g**0.36)
        Nusselt_air.append(Nu_g)
        print("Nusselt number for the air side:", np.round(Nu_g,4))

    h_g = (Nu_g * k_g)/D_o
    Heat_transfer_air.append(h_g)
    print("Heat transfer coefficient of the air side {} W/m\u00B2.K".format(np.round(h_g,4)))
    T = float(T_w)
    hwe = 2501 #(kJ/kg)
    h_fg = -0.0021*(float(T_g)**2) - 2.2115*(float(T_g)) + 2499
    Latent_heat_air.append(h_fg)
    print('Latent heat of vapourization {} Kj/kg'.format(np.round(h_fg,4)))
    # h_fg = 2207.7 # kj/kg
    D_h2oair = (6.057*(10**(-6)))               + (4.055*(10**(-8))*(float(T_g)+273))               + (1.25*(10**(-10))*(float(T_g)+273)**2)               - (3.367*(10**(-14))*(float(T_g)+273)**3)
    print("Mass diffusivity of water vapour in air: {} m\u00B2/s".format(D_h2oair))
    Mass_of_diffusivity.append(D_h2oair)
    D_h2og = D_h2oair*(alpha_g/(k_g_air/(rho_air*cp_Air*1000)))
    print("Mass diffusivity of water vapour in flue gas: {} m\u00B2/s".format(D_h2og))# 

    
    y_nb = 1 - y_h2o 
    Le_h20air = alpha_g / D_h2oair  
    Lewis_air.append(Le_h20air)
    print("Lewis number",np.round(Le_h20air,4))
    a = 16.262
    b = 3799.89
    c = 226.35
    T_g = float(T_g)
    h_g = h_g
    h_fg = h_fg
    y_h2o = y_h2o
    h_c = h_c
    T_c = T_c
    M_h2o = M_h2o
    c_pg = c_pg
    M_g = M_g
    p_tot = p/1000

    def equation(T_i):
        y_i = np.exp(a - (b / (T_i + c))) / p_tot
        y_nb = 1 - y_h2o 
        y_ni = 1-y_i 
        y_lm = (y_ni - y_nb) / math.log(y_ni / y_nb)
        Le_h20air = alpha_g / D_h2oair
        k_m = (h_g * M_h2o) / (c_pg*1000 * M_g * y_lm *Le_h20air ** (2/3))


        return ((h_g * T_g + h_fg*1000 * k_m * (y_h2o - y_i) + h_c * T_c) / (h_g + h_c)) - T_i
    if T_w<T_sat:
        print("There will be condensation")
        # Initial guess for T_i
        T_i_guess = 70
        T_i_solution = newton(equation, T_i_guess)
        T_i_guess = T_i_solution

        y_i = np.exp(a - (b / (T_i_solution + c))) / p_tot
        print("Interfacial mole fraction of water vapour {}".format(np.round(y_i,4)))
        y_nb = 1 - y_h2o 
        y_ni = 1-y_i 
        y_lm = (y_ni - y_nb) / math.log(y_ni / y_nb)

        
        Temperature_interface.append(T_i_solution)
        print("Newton-Raphson solution for temperature interface:", np.round(T_i_solution,4))
        Vapour_mole_interface.append(y_i)
        Le_h20air = alpha_g / D_h2oair
        k_m = (h_g * M_h2o) / (c_pg*1000 * M_g * y_lm *Le_h20air ** (2/3))
        Mass_transfer_coefficient_air.append(k_m)
        numbering.append(_+1)
        print("Mass transfer coefficient {}".format(np.round(k_m,4)))
        print("Lewis Number", Le_h20air)
        Logarithmic_mole_average.append(y_lm)
        print("Logarithmic average of non condensable gas at the interface", np.round(y_lm,4))
        print("Sum of number of moles at the interface",np.round(y_lm + y_i ),4)
    else:
        print("There is no condensation, therefore no interfacial parameters")
#         Temperature_interface.append(0)
#         Vapour_mole_interface.append(0)
#         Logarithmic_mole_average.append(0)


# In[5]:


Humid_air = df1.loc[e ,['Temperature decrease of the mixture_1',
 'Temperature decrease of the mixture_2',
 'Temperature decrease of the mixture_3',
 'Temperature decrease of the mixture_4',
 'Temperature decrease of the mixture_5',
 'Temperature decrease of the mixture_6',
 'Temperature decrease of the mixture_7',
 'Temperature decrease of the mixture_8']]

Cooling_water = df1.loc[e ,['Temperature increase of the cooling water_1',
 'Temperature increase of the cooling water_2',
 'Temperature increase of the cooling water_3',
 'Temperature increase of the cooling water_4',
 'Temperature increase of the cooling water_5',
 'Temperature increase of the cooling water_6',
 'Temperature increase of the cooling water_7',
 'Temperature increase of the cooling water_8']]

Wall_temp = df1.loc[e ,['Tube_coil_surface_temp1',
 'Tube_coil_surface_temp2',
 'Tube_coil_surface_temp3',
 'Tube_coil_surface_temp4',
 'Tube_coil_surface_temp5',
 'Tube_coil_surface_temp6',
 'Tube_coil_surface_temp7',
 'Tube_coil_surface_temp8']]

dew_point = df1.loc[e,['Dew point_1','Dew point_2','Dew point_3','Dew point_4','Dew point_5','Dew point_6',
'Dew point_7','Dew point_8']]

Nusselt = df1.loc[e,['First_Nusselt',
 'Second_Nusselt',
 'Third_Nusselt',
 'Fourth_Nusselt',
 'Fifth_Nusselt',
 'Sixth_Nusselt',
 'Seventh_Nusselt',
 'Eighth_Nusselt']]

Nusselt_conv = df2.loc[e,['Nusselt_1',
 'Nusselt_2',
 'Nusselt_3',
 'Nusselt_4',
 'Nusselt_5',
 'Nusselt_6',
 'Nusselt_7',
 'Nusselt_8']]

Condensation_rate = df1.loc[e,['First_Cond',
 'Second_Cond',
 'Third_Cond',
 'Fourth_Cond',
 'Fifth_Cond',
 'Sixth_Cond',
 'Seventh_Cond',
 'Eighth_Cond']]/df1.loc[e,'Time'] #g/s


# In[6]:


col1, col2= st.columns([28,1])
fig1= go.Figure()

fig1.add_trace(go.Scatter(x=numbering, y=Temperature_interface, mode='lines+markers', name='Temperature_interface', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=numbering, y=Humid_air, mode='lines+markers', name='Humid air', line=dict(color='red')))
fig1.add_trace(go.Scatter(x=numbering, y=Cooling_water, mode='lines+markers', name='Cooling water', line=dict(color='green')))
fig1.add_trace(go.Scatter(x=numbering, y=Wall_temp, mode='lines+markers', name='Wall temperature', line=dict(color='purple')))
fig1.add_trace(go.Scatter(x=numbering, y=dew_point, mode='lines+markers', name='Dew point temperature', line=dict(color='orange')))


fig1.update_layout(
    title="Temperature Profiles",
    xaxis_title="Local point",
    yaxis_title="Temperature (°C)",
    legend=dict(x=1.02, y=0.765),
    showlegend=True
)

col1.write(fig1)


# In[7]:


fig2= go.Figure()

fig2.add_trace(go.Scatter(x=numbering, y=Nusselt, mode='lines+markers', name='Overall Nusselt number', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=numbering, y=Nusselt_conv, mode='lines+markers', name='Convective Nusselt number', line=dict(color='orange')))

fig2.update_layout(
    title="Overall Nusselt number",
    xaxis_title="Local point",
    yaxis_title="Nusselt number",
    legend=dict(x=1.02, y=0.765),
    showlegend=True
)

st.write(fig2)


# In[8]:


cols = ['First_Cond','Second_Cond','Third_Cond','Fourth_Cond','Fifth_Cond','Sixth_Cond','Seventh_Cond','Eighth_Cond']
Condensation_rate['point_1'] = Condensation_rate[0] + Condensation_rate[1]
Condensation_rate['point_2'] = Condensation_rate[2] + Condensation_rate[3]
Condensation_rate['point_3'] = Condensation_rate[4] + Condensation_rate[5]
Condensation_rate['point_4'] = Condensation_rate[6] + Condensation_rate[7]


# Shorten column selection for clarity (optional)


# In[9]:


Condensation_rate = Condensation_rate.drop(labels=cols)


# In[10]:


x = np.arange(1, 5, 1)  # Condensation points
y = Condensation_rate  # Condensation rate values

# Create a trace for the bar chart
trace = go.Bar(
    x=x,
    y=y,
    name='Condensation Rate (g/s)',
    marker=dict(color='blue')# Add a name for the data series
)

# Create the figure layout
layout = go.Layout(
    title='Condensation Rate',  # Chart title
    xaxis_title='Condensation Point',  # Label for x-axis
    yaxis_title='Condensation Rate (g/s)',  # Label for y-axis
    xaxis=dict(  # Customize x-axis ticks (optional)
        tickmode='array',
        tickvals=x,
        ticktext=x
    ) # Enable grid lines
)

# Create the figure
fig3 = go.Figure(data=[trace], layout=layout)

# Display the chart
col2.write(fig3)


# In[11]:


Humid_air = df1.loc[e ,['Temperature decrease of the mixture_1',
 'Temperature decrease of the mixture_2',
 'Temperature decrease of the mixture_3',
 'Temperature decrease of the mixture_4',
 'Temperature decrease of the mixture_5',
 'Temperature decrease of the mixture_6',
 'Temperature decrease of the mixture_7',
 'Temperature decrease of the mixture_8']]

Cooling_water = df1.loc[e ,['Temperature increase of the cooling water_1',
 'Temperature increase of the cooling water_2',
 'Temperature increase of the cooling water_3',
 'Temperature increase of the cooling water_4',
 'Temperature increase of the cooling water_5',
 'Temperature increase of the cooling water_6',
 'Temperature increase of the cooling water_7',
 'Temperature increase of the cooling water_8']]

Wall_temp = df1.loc[e ,['Tube_coil_surface_temp1',
 'Tube_coil_surface_temp2',
 'Tube_coil_surface_temp3',
 'Tube_coil_surface_temp4',
 'Tube_coil_surface_temp5',
 'Tube_coil_surface_temp6',
 'Tube_coil_surface_temp7',
 'Tube_coil_surface_temp8']]

dew_point = df1.loc[e,['Dew point_1','Dew point_2','Dew point_3','Dew point_4','Dew point_5','Dew point_6',
'Dew point_7','Dew point_8']]


# In[12]:


# Combine the Series into a DataFrame
data = {
    'Humid_air': Humid_air.values,
    'Cooling_water': Cooling_water.values,
    'Wall_temp': Wall_temp.values,
    'Dew_point': dew_point.values,
    'Nusselt' : Nusselt.values,
    'Convective Nusselt': Nusselt_conv.values
}

combined_df = pd.DataFrame(data, index=Humid_air.index)

# Transpose the DataFrame
transposed_df = combined_df.T
transposed_df.columns = [f'point {i+1}' for i in range(len(transposed_df.columns))]


# In[13]:


check1 = st.checkbox('Click here to show the data')
if check1==True:
    st.write("The data is : ",transposed_df)


# In[ ]:





# In[ ]:





# In[ ]:




