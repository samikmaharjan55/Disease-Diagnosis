#================================================================================================================
#TITLE: THE FINAL MAMDANI MODEL FOR HEART DISEASE DIAGNOSIS
#Written by: Rupesh Kumar Dey / Loh Khai Ren / Rathi Tevi / Dharmendra
#TPNUM: TP061720 / TP062775 / TP061429 / TP061511
#Date: 20-Feb-2021
#================================================================================================================

#================================================================================================================
#================================================================================================================
#IMPORTING NECESSARY PYTHON PACKAGES
#================================================================================================================
#================================================================================================================
import pandas as pd
import Memberships as mm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#================================================================================================================
#================================================================================================================
#IMPORTING RAW & MANAGING DATASET
#================================================================================================================
#================================================================================================================
#IMPORTING DATA
df1 = pd.read_csv(r'D:\Rupesh\Documents\MASTERS IN AI\FUZZY Logic\00 ASSIGNMENT\python_codes\HUNGARIAN.csv') #DATASET CSV 1
df2 = pd.read_csv(r'D:\Rupesh\Documents\MASTERS IN AI\FUZZY Logic\00 ASSIGNMENT\python_codes\CLEVELAND.csv') #DATASET CSV 2
df3 = pd.read_csv(r'D:\Rupesh\Documents\MASTERS IN AI\FUZZY Logic\00 ASSIGNMENT\python_codes\SWITZERLAND.csv') #DATASET CSV 3
df = pd.concat([df1,df2,df3]) #MERGING 3 DIFFERENT DATASET
dfnew = df[["age","cp","trestbps","chol","fbs","target"]] #SELECTING DESIRED INPUTS AND OUTPUT
dfnew = dfnew.reset_index() #RESETTING INDEX OF DATA.
# print(dfnew.shape)
#================================================================================================================
#REMOVING MISSING VALUES
#ANY DATASET ROWS THAT HAVE VALUE "?" WILL BE REMOVED.
counter = 0
for index, row in dfnew.iterrows():
    determine = "?" in row.unique()
    if determine == True:
        counter+=1
        dfnew = dfnew.drop([index])

#================================================================================================================
#================================================================================================================
#IMPLEMENTING MAMDANI MODEL
#================================================================================================================
#================================================================================================================

#================================================================================================================
#DEFINING MEMBERSHIP FUNCTIONS
#================================================================================================================

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#CHEST PAIN
#DEFINING INPUT FOR CHEST PAIN FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
cpv = np.linspace(0,5,800)
cp11 = np.zeros_like(cpv)
cp21 = np.zeros_like(cpv)
cp31 = np.zeros_like(cpv)
cp41 = np.zeros_like(cpv)

#GENERATING MEMBERSHIP INPUT VALUES FOR CHEST PAIN USING SELECTED MEMBERSHIP
for i in range(len(cpv)):
    #SET 1
    cp11[i] = mm.tri(cpv[i],0.5,1,1.5)
    cp21[i] = mm.tri(cpv[i],1.5,2,2.5)
    cp31[i] = mm.tri(cpv[i],2.5,3,3.5)
    cp41[i] = mm.tri(cpv[i],3.5,4,4.5)

#cp_round FUNCTION used for squaring the triangle function
def cp_round (arr):
    arr2 = []
    for i in range(len(arr)):
        if arr[i] > 0 and arr[i]<=1:
            arr2.append(1)
        else:
            arr2.append(0)
    return arr2

#calling cp_round function for each category of chest pain
cp1r = cp_round(cp11)
cp2r = cp_round(cp21)
cp3r = cp_round(cp31)
cp4r = cp_round(cp41)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#CHOLETSEROL
#DEFINING INPUT FOR CHOLESTEROL FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
chol = np.linspace(0,600,1200)
low_chol_2 = np.zeros_like(chol)
med_chol_2 = np.zeros_like(chol)
high_chol_2 = np.zeros_like(chol)

#GENERATING MEMBERSHIP INPUT VALUES FOR CHOLESTEROL USING SET 2
for i in range(len(chol)):
    low_chol_2[i] = mm.sigmoid(chol[i],-0.5,176)
    med_chol_2[i] = mm.gaus(chol[i],219,10)
    high_chol_2[i] = mm.sigmoid(chol[i],0.5,252)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#BLOOD PRESSURE
#DEFINING INPUT FOR BLOOD PRESSURE FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
bp = np.linspace(0,200,800)
low_bp_2 = np.zeros_like(bp)
med_bp_2 = np.zeros_like(bp)
high_bp_2 = np.zeros_like(bp)

#GENERATING MEMBERSHIP INPUT VALUES FOR BLOOD PRESSURE USING SELECTED MEMBERSHIP
for i in range(len(bp)):
    #INPUT TYPE 2. SIGMOID DECREASING, GAUSSIAN AND SIGMOID INCREASING - SET 2
    low_bp_2[i] = mm.sigmoid(bp[i],-0.5,118)
    med_bp_2[i] = mm.gaus(bp[i],130,5)
    high_bp_2[i] = mm.sigmoid(bp[i],0.5,144)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#BLOOD SUGAR
#DEFINING INPUT FOR BLOOD SUGAR FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
bsugar = np.linspace(0,1.5,400)
high_bsugar1 = np.zeros_like(bsugar)
low_bsugar1 = np.zeros_like(bsugar)

#GENERATING MEMBERSHIP INPUT VALUES FOR BLOOD SUGAR USING SELECTED MEMBERSHIP
for i in range(len(bsugar)):
    #INPUT TYPE 1 INCREASING FUNCTION
    high_bsugar1[i]=mm.flat(bsugar[i],1)
    low_bsugar1[i]= mm.revflat(bsugar[i],1)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#AGE
#DEFINING INPUT FOR AGE FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
age = np.linspace(0,100,1200)
young_age3 = np.zeros_like(age)
mid_age3 = np.zeros_like(age)
old_age3 = np.zeros_like(age)

#GENERATING MEMBERSHIP INPUT VALUES FOR AGE USING SELECTED MEMBERSHIP
for i in range(len(age)):
    #INPUT TYPE 3, DECREASING, TRIANGLE AND INCREASING FUNCTION (WITH EXCEPTION)
    young_age3[i]=mm.dec(age[i],27,33)
    mid_age3[i]=mm.tri(age[i],33,39,44)
    old_age3[i] = mm.inc(age[i],44,51)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#GENERATING OUTPUT VALUES MEMBERSHIP
out_test = np.linspace(0,4,200)
low_risk = np.zeros_like(out_test)
med_risk = np.zeros_like(out_test)
high_risk = np.zeros_like(out_test)

for i in range(len(out_test)):
    low_risk[i] = mm.square(out_test[i],0,2.5)
    med_risk[i] = mm.square(out_test[i],2.5,3.5)
    high_risk[i] = mm.square(out_test[i],3.5,4)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#EVALUATING CHEST PAIN LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_cp = dfnew.cp.to_numpy().astype(int).tolist() #CHOLESTEROL DATA CONVERTED INTO LIST VARIABLE
target = dfnew.target.to_numpy().astype(int).tolist() #TARGET VALUE CONVERTED INTO LIST VARIABLE.

#function to regroup the output values in the dataset
def roundup(arr):
    for i in range(len(arr)):
        # print(output_3[i])
        if arr[i] < 2.5:
            arr[i] = 0
        elif arr[i]>= 2.5 and arr[i]< 3.5:
             arr[i] = 1
        elif arr[i]>= 3.5:
             arr[i] = 2
    return arr

target = roundup(target)


cpi1 = []
cpi2 = []
cpi3 = []
cpi4 = []

for i in range(len(input_cp)):
    cpi1.append(mm.tri(input_cp[i], 0.5, 1, 1.5))
    cpi2.append(mm.tri(input_cp[i], 1.5, 2, 2.5))
    cpi3.append(mm.tri(input_cp[i], 2.5, 3, 3.5))
    cpi4.append(mm.tri(input_cp[i], 3.5, 4, 4.5))

cpi1 = cp_round(cpi1)
cpi2 = cp_round(cpi2)
cpi3 = cp_round(cpi3)
cpi4 = cp_round(cpi4)
#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#EVALUATING CHOLESTEROL LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_chol = dfnew.chol.to_numpy().astype(int).tolist() #CHOLESTEROL DATA CONVERTED INTO LIST VARIABLE

#EMPTY LIST TO BE APPENDED TO LATER ON (1 2 AND 3 CORRESPOND TO THE DIFFERENT COMBINATIONS OF MEMBERSHIP FUCNTIONS AS EXPLAINED ABOVE)
in_low_chol_2 = []
in_med_chol_2 = []
in_high_chol_2 = []

#EVALUATING CHOLESTEROL LEVELS USING MEMBERSHIP FUNCTIONS PROPOSED
for i in range(len(input_chol)):
    in_low_chol_2.append(mm.sigmoid(input_chol[i], -0.5, 176))
    in_med_chol_2.append(mm.gaus(input_chol[i], 219,10))
    in_high_chol_2.append(mm.sigmoid(input_chol[i], 0.5, 252))
#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#EVALUATING BLOOD PRESSURE LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_bp = dfnew.trestbps.to_numpy().astype(int).tolist() #BLOOD PRESSURE DATA CONVERTED INTO LIST VARIABLE

in_low_bp_2 = []
in_med_bp_2 = []
in_high_bp_2 = []

#EVALUATING BLOOD PRESSURE LEVELS USING MEMBERSHIP FUNCTIONS SELECTED
for i in range(len(input_bp)):
    in_low_bp_2.append(mm.sigmoid(input_bp[i], -0.5, 118))
    in_med_bp_2.append(mm.gaus(input_bp[i], 130,5))
    in_high_bp_2.append(mm.sigmoid(input_bp[i], 0.5, 144))
#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#EVALUATING BLOOD SUGAR LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_bsugar = dfnew.fbs.to_numpy().astype(int).tolist() #BLOOD SUGAR DATA CONVERTED INTO LIST VARIABLE

#EMPTY LIST TO BE APPENDED TO LATER ON (1 2 AND 3 CORRESPOND TO THE DIFFERENT COMBINATIONS OF MEMBERSHIP FUCNTIONS AS EXPLAINED ABOVE)
in_high_bsugar1 = []
in_low_bsugar1 = []

#EVALUATING BLOOD SUGAR LEVELS USING MEMBERSHIP FUNCTIONS SELECTED
for i in range(len(input_bsugar)):

    in_high_bsugar1.append(mm.flat(input_bsugar[i],1))
    in_low_bsugar1.append(mm.revflat(input_bsugar[i],1))

a = []
for i in range(len(in_high_bsugar1)):
    if in_high_bsugar1[i] != 0:
        a.append(in_high_bsugar1[i])

for i in range(len(input_bsugar)):
    if input_bsugar[i] < 1:
        in_high_bsugar1[i] = 0
#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#EVALUATING AGE LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_age = dfnew.age.to_numpy().astype(int).tolist() #AGE DATA CONVERTED INTO LIST VARIABLE

#EMPTY LIST TO BE APPENDED TO LATER ON (1 2 AND 3 CORRESPOND TO THE DIFFERENT COMBINATIONS OF MEMBERSHIP FUCNTIONS AS EXPLAINED ABOVE)
in_young_age3 = []
in_mid_age3 = []
in_old_age3 = []

#EVALUATING AGE LEVELS USING MEMBERSHIP FUNCTIONS SELECTED
for i in range(len(input_age)):
    in_young_age3.append(mm.dec(input_age[i],27,33))
    in_mid_age3.append(mm.tri(input_age[i],33,39,44))
    in_old_age3.append(mm.inc(input_age[i],44,51))

#Exception for age
a = []
aa= []
aaa = []
for i in range(len(in_young_age3)):
    if in_young_age3[i] != 0:
        a.append(in_young_age3[i])
    elif in_mid_age3[i] != 0:
        aa.append(in_mid_age3[i])
    elif in_old_age3[i] != 0:
        aaa.append(in_old_age3[i])

for i in range(len(input_age)):
    if input_age[i] == 33:
        in_mid_age3[i] = 0.5 * (min(a) + min(aa))
    elif input_age[i] == 44:
        in_old_age3[i] = 0.5 * (min(aa) + min(aaa))
#================================================================================================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#================================================================================================================
#IMPLEMENTING RULES
#================================================================================================================

#CREATING EMPTY LISTS TO STORE OUTPUT VALUES CALCULATED BY MODEL
output_1 = []
output_2 = []
output_3 = []
output_4 = []
output_5 = []
output_6 = []
output_7 = []
output_8 = []
output_9 = []
output_10 = []
output_11 = []
output_chol = []
output_bp = []
output_cp = []
output_age = []
output_bsugar = []
output_final = []

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#DEFINING OUTCOME CASES
#RULE OUTCOME (THEN CONSEQUENCE / OUTCOME)
#np.maximum to be used if you want to change to UNION ie CONDITIONS IS OR
#np.minimum can be used if you want to change to INTERSECTION ie CONDITIONS IS AND

#-------------------------------------------------------------------------
# RO3 = np.maximum(high_risk,np.maximum(low_risk,med_risk)) #when considering 3 output class
# RO2 = np.maximum(low_risk,med_risk) #when considering 2 output class
# RO1 = low_risk or med_risk or high_risk #When considering 1 output class.
#-------------------------------------------------------------------------
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# RO1 = np.maximum(high_risk,med_risk)
# print(len(input_bp))

#Counters for checking purposes
#Counter for Checking Purposes
# count1 = 0;count2 = 0;count3 = 0;count4 = 0;count5 = 0;count6 = 0;count7 = 0;count8 = 0;count9 = 0;count10 = 0;count11 = 0;countfinal = 0

#Function for exception handling alternative 2
def next_small(arr):
    newarr=[]
    for i in range(len(arr)):
        if arr[i] != 0:
            newarr.append(arr[i])
    return (min(newarr))

#EVALUATING RULE VALUES BASED ON INPUT PARAMETERS FOR MEMBERSHIP FUCNTIONS:
for i in range(len(input_bp)):

    # RULE INPUT (IF IC - input condition)
    # np.minimum is considering that there is INTERSECTION ie CONDITIONS IS AND
    # np.maximum can be used if you want to change to UNION ie CONDITIONS IS OR

    # -------------------------------------------------------------------------
    # RI5 = RI5 = np.minimum(IC5[i],np.minimum(IC4[i],np.minimum(IC3[i],np.minimum(IC2[i],IC1[i])))) #when considering all 5 inputs
    # RI4 = np.minimum(IC4[i],np.minimum(IC3[i],np.minimum(IC2[i],IC1[i]))) #when considering all 4 inputs
    # RI3 = np.minimum(IC3[i],np.minimum(IC2[i],IC1[i])) #when considering all 3 inputs
    # -------------------------------------------------------------------------

    #
    # RULE 1
    cage1 = np.maximum(in_mid_age3[i],in_old_age3[i])
    ccp1 = cpi1[i]
    cbp1 = in_high_bp_2[i]
    cch1 = in_high_chol_2[i]
    RI1 = np.minimum(cage1,np.minimum(cpi1[i],np.maximum(cbp1,cch1)))
    # if RI1.min() == 0:
    #     mem_RI1 = [cage1,ccp1,cbp1,cch1]
    #     # print(mem_RI1)
    #     RI1 = next_small(mem_RI1)
    #     # print(RI1)
    RO1 = np.maximum(high_risk, med_risk)

    #RULE 2
    cage2 = np.maximum(in_mid_age3[i],in_young_age3[i])
    ccp2 = cpi4[i]
    cbp2 = np.maximum(in_low_bp_2[i],in_med_bp_2[i])
    cch2 = np.maximum(in_low_chol_2[i], in_med_chol_2[i])
    RI2 = np.minimum(cage2,np.minimum(ccp2,np.maximum(cbp2,cch2)))
    # if RI2.min() == 0:
    #     mem_RI2 = [cage2,ccp2,cbp2,cch2]
    #     # print(mem_RI2)
    #     RI2 = next_small(mem_RI2)
    RO2 = low_risk

    #RULE 3
    cage3 = in_young_age3[i]
    csug3 = in_high_bsugar1[i]
    ccp3 = np.maximum(cpi1[i],cpi2[i])
    cbp3 = in_high_bp_2[i]
    cch3 = in_high_chol_2[i]
    RI3 = np.minimum(ccp3,np.minimum(cage3,np.maximum(csug3,np.minimum(cbp3,cch3))))
    # if RI3.min() == 0:
    #     mem_RI3 = [cage3,ccp3,cbp3,cch3,csug3]
    #     # print(mem_RI3)
    #     RI3 = next_small(mem_RI3)
    RO3 = high_risk

    #RULE 4
    cage4 = in_old_age3[i]
    csug4 = in_low_bsugar1[i]
    ccp4 = cpi4[i]
    cbp4 = in_med_bp_2[i]
    cch4 = in_low_chol_2[i]
    RI4 = np.minimum(ccp4,np.minimum(cage4,np.maximum(csug4,np.minimum(cbp4,cch4))))
    # if RI4.min() == 0:
    #     mem_RI4 = [cage4,ccp4,cbp4,cch4,csug4]
    #     # print(mem_RI4)
    #     RI4 = next_small(mem_RI4)
    RO4 = low_risk

    #RULE 5
    # IF (chest pain 3) AND (middle) AND (high sugar) OR (low BP AND medium Chol) THEN (medium Risk)
    cage5 = in_mid_age3[i]
    csug5 = in_high_bsugar1[i]
    ccp5 = cpi3[i]
    cbp5 = in_low_bp_2[i]
    cch5 = in_med_chol_2[i]
    RI5 = np.minimum(ccp5,np.minimum(cage5,np.maximum(csug5,np.minimum(cbp5,cch5))))
    # if RI5.min() == 0:
    #     mem_RI5 = [cage5,ccp5,cbp5,cch5,csug5]
    #     # print(mem_RI5)
    #     RI5 = next_small(mem_RI5)
    RO5 = med_risk

    #RULE 6
    # IF (chest pain 2) AND (middle) AND (low sugar) OR (high BP AND high Chol) THEN (medium Risk)
    cage6 = in_mid_age3[i]
    csug6 = in_low_bsugar1[i]
    ccp6 = cpi2[i]
    cbp6 = in_high_bp_2[i]
    cch6 = in_high_chol_2[i]
    RI6 = np.minimum(ccp6,np.minimum(cage6,np.minimum(csug6,np.minimum(cbp6,cch6))))
    # if RI6.min() == 0:
    #     mem_RI6 = [cage6,ccp6,cbp6,cch6,csug6]
    #     # print(mem_RI6)
    #     RI6 = next_small(mem_RI6)
    RO6 = med_risk

    #RULE 7
    # IF (chest pain 1) AND (old) AND (high sugar) OR (medium BP AND high Chol) THEN (high Risk)
    cage7 = in_old_age3[i]
    csug7 = in_high_bsugar1[i]
    ccp7 = cpi1[i]
    cbp7 = in_med_bp_2[i]
    cch7 = in_high_chol_2[i]
    RI7 = np.minimum(ccp7,np.minimum(cage7,np.minimum(csug7,np.maximum(cbp7,cch7))))
    # if RI7.min() == 0:
    #     mem_RI7 = [cage7, ccp7, cbp7, cch7, csug7]
    #     # print(mem_RI7)
    #     RI7 = next_small(mem_RI7)
    RO7 = high_risk

    #RULE 8
    # IF (chest pain 2) AND (young) AND (low sugar) OR (high BP AND medium Chol) THEN (high Risk)
    cage8 = in_young_age3[i]
    csug8 = in_low_bsugar1[i]
    ccp8 = cpi2[i]
    cbp8 = in_high_bp_2[i]
    cch8 = in_med_chol_2[i]
    RI8 = np.minimum(ccp8,np.minimum(cage8,np.minimum(csug8,np.minimum(cbp8,cch8))))
    # if RI8.min() == 0:
    #     mem_RI8 = [cage8, ccp8, cbp8, cch8, csug8]
    #     # print(mem_RI8)
    #     RI8 = next_small(mem_RI8)
    RO8 = high_risk

    #RULE 9
    # IF (chest pain 1) AND (old) AND (low sugar) OR (low BP AND low Chol) THEN (medium Risk)
    cage9 = in_old_age3[i]
    csug9 = in_low_bsugar1[i]
    ccp9 = cpi1[i]
    cbp9 = in_med_bp_2[i]
    cch9 = in_low_chol_2[i]
    RI9 = np.minimum(ccp9,np.minimum(cage9,np.minimum(csug9,np.maximum(cbp9,cch9))))
    RO9 = high_risk

    #RULE 10
    # IF (chest pain 4) AND (old) AND (low sugar) OR (low BP AND low Chol) THEN (medium Risk)
    cage10 = in_old_age3[i]
    csug10 = in_low_bsugar1[i]
    ccp10 = cpi4[i]
    cbp10 = in_low_bp_2[i]
    cch10 = in_low_chol_2[i]
    RI10 = np.minimum(ccp10,np.minimum(cage10,np.minimum(csug10,np.maximum(cbp10,cch10))))
    RO10 = med_risk

    #RULE 11
    # IF (chest pain 3 or 4) AND (old) AND (low sugar) OR (low BP AND high Chol) THEN (high Risk)
    cage11 = in_old_age3[i]
    csug11 = in_high_bsugar1[i]
    ccp11 = np.maximum(cpi3[i],cpi4[i])
    cbp11 = in_low_bp_2[i]
    cch11 = in_high_chol_2[i]
    RI11 = np.minimum(ccp11,np.minimum(cage11,np.maximum(csug11,np.minimum(cbp11,cch11))))
    RO11 = high_risk

    # Chest Pain
    R_cp1 = np.fmin(cpi4[i], low_risk)
    R_cp2 = np.fmin(cpi3[i], low_risk)
    R_cp3 = np.fmin(cpi2[i], np.maximum(med_risk,high_risk))
    R_cp4 = np.fmin(cpi1[i], np.maximum(med_risk,high_risk))
    R_cp = np.maximum(R_cp4, np.maximum(R_cp3, np.maximum(R_cp2, R_cp1)))
    outputcp = np.trapz(R_cp * out_test, out_test) / np.trapz(R_cp, out_test)  # calculating centroid for union of rules.
    if np.isnan(outputcp):
        outputcp = 0
    output_cp.append(outputcp)

    #Simple rules from input membership definition
    #Cholesterol
    R_chol1 = np.fmin(in_low_chol_2[i], low_risk)
    R_chol2 = np.fmin(in_med_chol_2[i], low_risk)
    R_chol3 = np.fmin(in_high_chol_2[i], np.maximum(med_risk,high_risk))
    R_chol = np.maximum(R_chol3, np.maximum(R_chol2, R_chol1))
    outputchol = np.trapz(R_chol * out_test, out_test) / np.trapz(R_chol, out_test)  # calculating centroid for union of rules.
    if np.isnan(outputchol):
        outputchol = 0
    output_chol.append(outputchol)

    #Blood pressure
    R_bp1 = np.fmin(in_low_bp_2[i], low_risk)
    R_bp2 = np.fmin(in_med_bp_2[i], low_risk)
    R_bp3 = np.fmin(in_high_bp_2[i], np.maximum(med_risk,high_risk))
    R_bp = np.maximum(R_bp3, np.maximum(R_bp2, R_bp1))
    outputbp = np.trapz(R_bp * out_test, out_test) / np.trapz(R_bp, out_test)  # calculating centroid for union of rules.
    if np.isnan(outputbp):
        outputbp = 0
    output_bp.append(outputbp)

    #Age
    R_age1 = np.fmin(in_young_age3[i], low_risk)
    R_age2 = np.fmin(in_mid_age3[i], low_risk)
    R_age3 = np.fmin(in_old_age3[i], np.maximum(med_risk, high_risk))
    R_age = np.maximum(R_age3, np.maximum(R_age2, R_age1))
    outputage = np.trapz(R_age * out_test, out_test) / np.trapz(R_age,out_test)  # calculating centroid for union of rules.
    if np.isnan(outputage):
        outputage = 0
    output_age.append(outputage)

    #Diabetes
    R_bsugar1 = np.fmin(in_low_bsugar1[i], low_risk)
    R_bsugar2 = np.fmin(in_high_bsugar1[i], high_risk)
    R_bsugar = np.maximum(R_bsugar2, R_bsugar1)
    outputbsugar = np.trapz(R_bsugar * out_test, out_test) / np.trapz(R_bsugar,out_test)  # calculating centroid for union of rules.
    if np.isnan(outputbsugar):
        outputbsugar = 0
    output_bsugar.append(outputbsugar)


    #Compoisition of rules
    R1 = np.fmin(RI1,RO1)
    R2 = np.fmin(RI2,RO2)
    R3 = np.fmin(RI3,RO3)
    R4 = np.fmin(RI4,RO4)
    R5 = np.fmin(RI5,RO5)
    R6 = np.fmin(RI6,RO6)
    R7 = np.fmin(RI7,RO7)
    R8 = np.fmin(RI8,RO8)
    R9 = np.fmin(RI9, RO9)
    R10 = np.fmin(RI10, RO10)
    R11 = np.fmin(RI11, RO11)

    #Aggregation of rules for input
    R_inputs = np.maximum(R_age, np.maximum(R_bsugar, np.maximum(R_cp, np.maximum(R_chol, R_bp))))

    # Aggregation of 8 rules + Rules for inputs
    R = np.maximum(R_inputs,np.maximum(R9,np.maximum(R8,np.maximum(R7,np.maximum(R6,np.maximum(R5,np.maximum(R4,np.maximum(R3,np.maximum(R2,R1)))))))))
    # print(R)

    # output1 = np.trapz(R1 * out_test, out_test) / np.trapz(R1,out_test)  # calculating centroid for union of rules.
    # output2 = np.trapz(R2 * out_test, out_test) / np.trapz(R2,out_test)  # calculating centroid for union of rules.
    # output3 = np.trapz(R3 * out_test, out_test) / np.trapz(R3, out_test)  # calculating centroid for union of rules.
    # output4 = np.trapz(R4 * out_test, out_test) / np.trapz(R4, out_test)  # calculating centroid for union of rules.
    # output5 = np.trapz(R5 * out_test, out_test) / np.trapz(R5, out_test)  # calculating centroid for union of rules.
    # output6 = np.trapz(R6 * out_test, out_test) / np.trapz(R6, out_test)  # calculating centroid for union of rules.
    # output7 = np.trapz(R7 * out_test, out_test) / np.trapz(R7, out_test)  # calculating centroid for union of rules.
    # output8 = np.trapz(R8 * out_test, out_test) / np.trapz(R8, out_test)  # calculating centroid for union of rules.
    # output9 = np.trapz(R9 * out_test, out_test) / np.trapz(R9, out_test)  # calculating centroid for union of rules.
    # output10 = np.trapz(R10 * out_test, out_test) / np.trapz(R10, out_test)  # calculating centroid for union of rules.
    # output11 = np.trapz(R11 * out_test, out_test) / np.trapz(R11, out_test)  # calculating centroid for union of rules.
    #Calculate the final deffuzified output
    outputfinal = np.trapz(R * out_test, out_test) / np.trapz(R, out_test)  # calculating centroid for union of rules.

    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output1):
    #     # print("ERROR")
    #     count1 += 1
    #     # print(count)
    #     # print("Age:",input_age[i]," Cholesterol",input_chol[i]," BP",input_bp[i]," Sugar",input_bsugar[i]," CP",input_cp[i])
    #     output1 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_1.append(output1)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output2):
    #     # print("ERROR")
    #     count2 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output2 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_2.append(output2)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output3):
    #     # print("ERROR")
    #     count3 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output3 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_3.append(output3)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output4):
    #     # print("ERROR")
    #     count4 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output4 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_4.append(output4)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output5):
    #     # print("ERROR")
    #     count5 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output5 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_5.append(output5)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output6):
    #     # print("ERROR")
    #     count6 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output6 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_6.append(output6)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output7):
    #     # print("ERROR")
    #     count7 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output7 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_7.append(output7)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output8):
    #     # print("ERROR")
    #     count8 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output8 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_8.append(output8)
    #
    # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output9):
    #     # print("ERROR")
    #     # count9 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output9 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_9.append(output9)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output10):
    #     # print("ERROR")
    #     count10 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output10 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_10.append(output10)
    #
    # # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # # ERROR MESSAGE WILL BE PRINTED
    # if np.isnan(output11):
    #     # print("ERROR")
    #     count11 += 1
    #     # print(count)
    #     # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
    #     output11 = 0
    # # APPEND TO THE RESPECTIVE OUTPUT LIST
    # output_11.append(output11)

    # TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    # ERROR MESSAGE WILL BE PR

    if np.isnan(outputfinal):
        print("ERROR")
        # countfinal += 1
        # print(count)
        # print("Age:", input_age[i], " Cholesterol", input_chol[i], " BP", input_bp[i], " Sugar", input_bsugar[i], " CP",input_cp[i])
        # print(RI1,RI2,RI3,RI4,RI5,RI6,RI7,RI8)
        outputfinal = 0
    # APPEND TO THE RESPECTIVE OUTPUT LIST
    output_final.append(outputfinal)

    #For checking
    # print("Count 1",count1,"Count 2",count2,"Count 3",count3,"Count 4",count4,"Count 5",count5,"Count 6",count6,"Count 7",count7,"Count 8",count8,"Count 9",count9, "Count 10",count10,"Count 11",count11,"Count final",countfinal,)


# output_1 = roundup(output_1)
# output_2 = roundup(output_2)
# output_3 = roundup(output_3)
# output_4 = roundup(output_4)
# output_5 = roundup(output_5)
# output_6 = roundup(output_6)
# output_7 = roundup(output_7)
# output_8 = roundup(output_8)
# output_9 = roundup(output_9)
# output_10 = roundup(output_10)
# output_11 = roundup(output_11)

#Categorizing the final calculated deffuzified output
output_final = roundup(output_final)

#================================================================================================================
#CALCULATING ACCURACY AND CONFUSION MATRIX
#================================================================================================================

# ## EVALUATING CONFUSION MATRIX AND ACCURACY
# cf1 = confusion_matrix(target,output_1)
# acc1 = accuracy_score(target,output_1)
# print("accuracy output 1:",acc1*100)
#
# # USE AS REQUIRED
# cf2 = confusion_matrix(target,output_2)
# acc2 = accuracy_score(target,output_2)
# print("accuracy output 2:",acc2*100)
#
# cf3 = confusion_matrix(target,output_3)
# acc3 = accuracy_score(target,output_3)
# print("accuracy output 3:",acc3*100)
#
# cf4 = confusion_matrix(target,output_4)
# acc4 = accuracy_score(target,output_4)
# print("accuracy output 4:",acc4*100)
#
# cf5 = confusion_matrix(target,output_5)
# acc5 = accuracy_score(target,output_5)
# print("accuracy output 5:",acc5*100)
#
# cf6 = confusion_matrix(target,output_6)
# acc6 = accuracy_score(target,output_6)
# print("accuracy output 6:",acc6*100)
#
# cf7 = confusion_matrix(target,output_7)
# acc7 = accuracy_score(target,output_7)
# print("accuracy output 7:",acc7*100)
#
# cf8 = confusion_matrix(target,output_8)
# acc8 = accuracy_score(target,output_8)
# print("accuracy output 8:",acc8*100)
#
# cf9 = confusion_matrix(target,output_9)
# acc9 = accuracy_score(target,output_9)
# print("accuracy output 9:",acc9*100)
#
# cf10 = confusion_matrix(target,output_10)
# acc10 = accuracy_score(target,output_10)
# print("accuracy output 10:",acc10*100)
#
# cf11 = confusion_matrix(target,output_11)
# acc11 = accuracy_score(target,output_11)
# print("accuracy output 11:",acc11*100)

#Calculating accuracy and confusion matrix
cffinal = confusion_matrix(target,output_final)
accfinal = accuracy_score(target,output_final)
print("accuracy output Final:",accfinal*100)


#USE AS REQUIRED
# cf2 = confusion_matrix(target,output_2)
# acc2 = accuracy_score(target,output_2)
# print("accuracy output 1:",acc1*100)

plt.figure()
plt.title("Output Membership Functions - Regrouped")
plt.xlabel("Category of output")
plt.ylabel("Fuzzy output")
plt.plot(out_test,low_risk,label = "Low Risk")
plt.plot(out_test,med_risk,label = "Medium Risk")
plt.plot(out_test,high_risk,label = "High Risk")
# plt.plot(out_test,R1,label="Rule -1")
# plt.plot(out_test,R_inputs,label="Rule -inputs")
# plt.plot(out_test,R,label="Aggregate Rule")

plt.legend()
plt.show()

