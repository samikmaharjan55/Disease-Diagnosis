#================================================================================================================
#TITLE: DETERMINING MEMBERSHIP FUNCTION FOR BLOOD PRESSURE
#Written by: Rupesh Kumar Dey
#TPNUM: TP061720
#Date: 14-Feb-2021
#================================================================================================================
#IMPORTING NECESSARY PYTHON PACKAGES
import pandas as pd
import Memberships as mm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#================================================================================================================
#IMPORTING RAW DATASET
df1 = pd.read_csv(r'D:\Rupesh\Documents\MASTERS IN AI\FUZZY Logic\00 ASSIGNMENT\python_codes\HUNGARIAN.csv') #DATASET CSV 1
df2 = pd.read_csv(r'D:\Rupesh\Documents\MASTERS IN AI\FUZZY Logic\00 ASSIGNMENT\python_codes\CLEVELAND.csv') #DATASET CSV 2
df3 = pd.read_csv(r'D:\Rupesh\Documents\MASTERS IN AI\FUZZY Logic\00 ASSIGNMENT\python_codes\SWITZERLAND.csv') #DATASET CSV 3
df = pd.concat([df1,df2,df3]) #MERGING 3 DIFFERENT DATASET
dfnew = df[["age","cp","trestbps","chol","fbs","target"]] #SELECTING DESIRED INPUTS AND OUTPUT
dfnew = dfnew.reset_index() #RESETTING INDEX OF DATA.
#================================================================================================================
#REMOVING MISSING VALUES
#ANY DATASET ROWS THAT HAVE VALUE "?" WILL BE REMOVED.
counter = 0
for index, row in dfnew.iterrows():
    determine = "?" in row.unique()
    if determine == True:
        counter+=1
        dfnew = dfnew.drop([index])

print(type(dfnew))

#================================================================================================================
#MAMDANI METHOD
#DEFINING INPUT FOR BLOOD PRESSURE FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
bp = np.linspace(0,200,800)
low_bp_1 = np.zeros_like(bp)
med_bp_1 = np.zeros_like(bp)
high_bp_1 = np.zeros_like(bp)

low_bp_2 = np.zeros_like(bp)
med_bp_2 = np.zeros_like(bp)
high_bp_2 = np.zeros_like(bp)

low_bp_3 = np.zeros_like(bp)
med_bp_3 = np.zeros_like(bp)
high_bp_3 = np.zeros_like(bp)

#================================================================================================================
#GENERATING MEMBERSHIP INPUT VALUES FOR BLOOD PRESSURE USING SELECTED MEMBERSHIP
for i in range(len(bp)):
    #INPUT TYPE 1 DECREASING, TRIANGLE AND INCREASING FUNCTION - SET1
    low_bp_1[i] = mm.dec(bp[i],111,125)
    med_bp_1[i] = mm.tri(bp[i],120,130,140)
    high_bp_1[i] = mm.inc(bp[i],130,157)

    #INPUT TYPE 2. SIGMOID DECREASING, GAUSSIAN AND SIGMOID INCREASING - SET 2
    low_bp_2[i] = mm.sigmoid(bp[i],-0.5,118)
    med_bp_2[i] = mm.gaus(bp[i],130,5)
    high_bp_2[i] = mm.sigmoid(bp[i],0.5,144)

    #INPUT TYPE 3, DECREASING, GAUSSIAN AND INCREASING FUNCTION - SET 3
    low_bp_3[i] = mm.z_func_dec(bp[i],111,125)
    med_bp_3[i] = mm.gaus(bp[i],130,5)
    high_bp_3[i] = mm.z_func_inc(bp[i],130,157)

#================================================================================================================
#GENERATING OUTPUT VALUES MEMBERSHIP
out_test = np.linspace(0,4,200)
healthy = np.zeros_like(out_test)
sick1 = np.zeros_like(out_test)
sick2 = np.zeros_like(out_test)
sick3 = np.zeros_like(out_test)
sick4 = np.zeros_like(out_test)

for i in range(len(out_test)):
    healthy[i] = mm.dec(out_test[i],0.25,0.5)
    sick1[i] = mm.tri(out_test[i],0.5,1,1.5)
    sick2[i] = mm.tri(out_test[i],1.5,2,2.5)
    sick3[i] = mm.tri(out_test[i],2.5,3,3.5)
    sick4[i] = mm.inc(out_test[i],3.5,3.75)

#================================================================================================================
#EVALUATING BLOOD PRESSURE LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_bp = dfnew.trestbps.to_numpy().astype(int).tolist() #BLOOD PRESSURE DATA CONVERTED INTO LIST VARIABLE
target = dfnew.target.to_numpy().astype(int).tolist() #TARGET VALUE CONVERTED INTO LIST VARIABLE.

#EMPTY LIST TO BE APPENDED TO LATER ON (1 2 AND 3 CORRESPOND TO THE DIFFERENT COMBINATIONS OF MEMBERSHIP FUCNTIONS AS EXPLAINED ABOVE)
in_low_bp_1 = []
in_med_bp_1 = []
in_high_bp_1 = []
#
in_low_bp_2 = []
in_med_bp_2 = []
in_high_bp_2 = []
#
in_low_bp_3 = []
in_med_bp_3 = []
in_high_bp_3 = []

#Filtering dataset based on ranges for each Fuzzy class
def segbp (arr1,arr2,val1,val2):
    o0 = 0
    o1 = 0
    o2 = 0
    o3 = 0
    o4 = 0
    for i in range(len(arr1)):
        if arr1[i] < val2 and arr1[i]>=val1:
            if arr2[i] ==0:
                o0+=1
            elif arr2[i] ==1:
                o1+=1
            elif arr2[i] == 2:
                o2 += 1
            elif arr2[i] ==3:
                o3+=1
            elif arr2[i] ==4:
                o4+=1
    return [o0,o1,o2,o3,o4]

cat1 = segbp(input_bp,target,0,125)
cat2 = segbp(input_bp,target,125,130)
cat3 = segbp(input_bp,target,130,1000)


#EVALUATING BLOOD PRESSURE LEVELS USING MEMBERSHIP FUNCTIONS SELECTED
for i in range(len(input_bp)):
    in_low_bp_1.append(mm.dec(input_bp[i], 111, 125))
    in_med_bp_1.append(mm.tri(input_bp[i], 120, 130, 140))
    in_high_bp_1.append(mm.inc(input_bp[i], 130, 157))

    in_low_bp_2.append(mm.sigmoid(input_bp[i], -0.5, 118))
    in_med_bp_2.append(mm.gaus(input_bp[i], 130,5))
    in_high_bp_2.append(mm.sigmoid(input_bp[i], 0.5, 144))
    #
    in_low_bp_3.append(mm.z_func_dec(input_bp[i], 111, 125))
    in_med_bp_3.append(mm.gaus(input_bp[i], 130, 5))
    in_high_bp_3.append(mm.z_func_inc(input_bp[i], 130, 157))

#================================================================================================================
#CREATING AN EMPTY OUTPUT ARRAY TO BE APPENDED TO LATER.
output_1 = []
output_2 = []
output_3 = []

##================================================================================================================
#GENERATING RULES
#Rules
#CREATING CASES TO BE USED IN RULES (COMBINATIONS OF HEART DISEASE RISK)
case1 = np.maximum(healthy,sick1)
case2 = np.maximum(sick1,sick2)
case3 = np.maximum(sick3,sick4)

#EVALUATING RULE VALUES BASED ON INPUT PARAMETERS FOR MEMBERSIP FUCNTION COMBINDATIONS 1 2 AND 3
for i in range(len(input_bp)):
    #R11 - RULE 1 SET 1
    R11 = np.fmin(in_low_bp_1[i],case1)

    #R21 - RULE 2 FOR SET 1
    R21 = np.fmin(in_med_bp_1[i],case2)

    #RULE 31 - RULE 3 FOR SET 1
    R31 = np.fmin(in_high_bp_1[i],case3)

    #UNION OF RULES FOR SET 1
    R_1 = np.maximum(R31,np.maximum(R21,R11))
    output1 = np.trapz(R_1 * out_test, out_test) / np.trapz(R_1,out_test)  # calculating centroid for union of rules.
    #TO CHECK IF THERE'S AN ERROR. IF THERE IS AN ERROR THEN THE OUTPUT = 0
    if np.isnan(output1):
        output1 = 0
    #APPEND TO THE RESPECTIVE OUTPUT LIST
    output_1.append(output1)

    #SET 2
    R12 = np.fmin(in_low_bp_2[i], case1)
    R22 = np.fmin(in_med_bp_2[i], case2)
    R32 = np.fmin(in_high_bp_2[i], case3)
    R_2 = np.maximum(R32, np.maximum(R22, R12))
    output2 = np.trapz(R_2 * out_test, out_test) / np.trapz(R_2, out_test)  # calculating centroid for union of rules.
    if np.isnan(output2):
        output2 = 0
    output_2.append(output2)

    #SET 3
    R13 = np.fmin(in_low_bp_3[i], case1)
    R23 = np.fmin(in_med_bp_3[i], case2)
    R33 = np.fmin(in_high_bp_3[i], case3)
    R_3 = np.maximum(R33, np.maximum(R23, R13))
    output3 = np.trapz(R_3 * out_test, out_test) / np.trapz(R_3, out_test)  # calculating centroid for union of rules.
    if np.isnan(output3):
        output1 = 0
    output_3.append(output3)

#CATEGORIZING THE OUTPUT RESPECTIVE INITIAL GROUPS OF HEART DISEASE RISK
#SET 1
for i in range(len(output_1)):
    # print(output_1[i])
    if output_1[i] < 0.5:
        output_1[i] = 0
    elif (output_1[i] >=0.5) and (output_1[i] <1.5):
         output_1[i] = 1
    elif (output_1[i]>= 1.5) and (output_1[i] <2.5):
         output_1[i] = 2
    elif (output_1[i] >= 2.5) and (output_1[i] < 3.5):
        output_1[i] = 3
    elif output_1[i]>= 3.5:
         output_1[i] = 4
    # print(output_1[i])
# print(output_1)

#SET 2
for i in range(len(output_2)):
    # print(output_2[i])
    if output_2[i] < 0.5:
        output_2[i] = 0
    elif (output_2[i] >=0.5) and (output_2[i] <1.5):
         output_2[i] = 1
    elif (output_2[i]>= 1.5) and (output_2[i] <2.5):
         output_2[i] = 2
    elif (output_2[i] >= 2.5) and (output_2[i] < 3.5):
        output_2[i] = 3
    elif output_2[i]>= 3.5:
         output_2[i] = 4
    # print(output_2[i])
# print(output_2)

#SET 3
for i in range(len(output_3)):
    # print(output_3[i])
    if output_3[i] < 0.5:
        output_3[i] = 0
    elif (output_3[i] >=0.5) and (output_3[i] <1.5):
         output_3[i] = 1
    elif (output_3[i]>= 1.5) and (output_3[i] <2.5):
         output_3[i] = 2
    elif (output_3[i] >= 2.5) and (output_3[i] < 3.5):
        output_3[i] = 3
    elif output_3[i]>= 3.5:
         output_3[i] = 4
    # print(output_3[i])
# print(output_3)

# FUNCTION FOR REGROUPING GROUPS 0,1,2,3,4 INTO GROUPS OF 0,1,2
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

#REGROUPING OUTPUT DATA FOR TARGET AND PREDICTED OUTPUT
tround = roundup(target)
outround1 = roundup(output_1)
outround2 = roundup(output_2)
outround3 = roundup(output_3)

## EVALUATING CONFUSION MATRIX AND ACCURACY
cf1 = confusion_matrix(target,output_1)
acc1 = accuracy_score(target,output_1)

cf2 = confusion_matrix(target,output_2)
acc2 = accuracy_score(target,output_2)

cf3 = confusion_matrix(target,output_3)
acc3 = accuracy_score(target,output_3)

print("accuracy output 1:",acc1*100,"%","accuracy output 2:",acc2*100,"%","accuracy output 3:",acc3*100,"%")

##================================================================================================================
#PLOTTING DATA
plt.figure("Blood Pressure Membership Function Selection")
plt.subplot(2,3,1)
plt.title("Blood Pressure Linear - Set 1")
plt.xlabel("Blood Presure (mmHg)")
plt.ylabel("Fuzzy output")
plt.plot(bp,low_bp_1,label="Low BP")
plt.plot(bp,med_bp_1,label="Medium BP")
plt.plot(bp,high_bp_1,label="High BP")
plt.legend()

plt.subplot(2,3,2)
plt.title("Blood Pressure Sigmoid+Gauss - Set 2")
plt.xlabel("Blood Presure (mmHg)")
plt.ylabel("Fuzzy output")
plt.plot(bp,low_bp_2,label="Low BP")
plt.plot(bp,med_bp_2,label="Medium BP")
plt.plot(bp,high_bp_2,label="High BP")
plt.legend()

plt.subplot(2,3,3)
plt.title("Blood Pressure Z+Gauss - Set 3")
plt.xlabel("Blood Presure (mmHg)")
plt.ylabel("Fuzzy output")
plt.plot(bp,low_bp_3,label="Low BP")
plt.plot(bp,med_bp_3,label="Medium BP")
plt.plot(bp,high_bp_3,label="High BP")
plt.legend()

plt.subplot(2,3,4)
plt.title("Output Membership Functions")
plt.xlabel("Category of output")
plt.ylabel("Fuzzy output")
plt.plot(out_test,healthy,label="Healthy")
plt.plot(out_test,sick1,label="Sick1")
plt.plot(out_test,sick2,label="Sick2")
plt.plot(out_test,sick3,label="Sick3")
plt.plot(out_test,sick4,label="sick4")
plt.legend()

plt.subplot(2,3,5)
plt.title("Comparison of predicted output vs Target output")
plt.xlabel("Blood Presure (mmHg)")
plt.ylabel("Fuzzy output")
plt.plot(bp,low_bp_1,label="Low BP Set 1")
plt.plot(bp,med_bp_1,label="Medium BP Set 1")
plt.plot(bp,high_bp_1,label="High BP Set 1")
plt.plot(bp,low_bp_2,label="Low BP Set 2")
plt.plot(bp,med_bp_2,label="Medium BP Set 2")
plt.plot(bp,high_bp_2,label="High BP Set 2")
plt.plot(bp,low_bp_3,label="Low BP Set 3")
plt.plot(bp,med_bp_3,label="Medium BP Set 3")
plt.plot(bp,high_bp_3,label="High BP Set 3")
plt.legend()

plt.subplot(2,3,6)
w=0.1
x = ["Low","Medium","High"]
h = [cat1[0],cat2[0],cat3[0]]
s1 = [cat1[1],cat2[1],cat3[1]]
s2 = [cat1[2],cat2[2],cat3[2]]
s3 = [cat1[3],cat2[3],cat3[3]]
s4 = [cat1[4],cat2[4],cat3[4]]
st = h+s1+s2+s3+s4

# print(h)
# print(s1)
# print(s2)
# print(s3)
# print(s4)

bar1 = np.arange(len(x))
bar2 = [i+w for i in bar1]
bar3 = [i+2*w for i in bar1]
bar4 = [i+3*w for i in bar1]
bar5 = [i+4*w for i in bar1]
bart = [bar1[0],bar1[1],bar1[2],bar2[0],bar2[1],bar2[2],bar3[0],bar3[1],bar3[2],bar4[0],bar4[1],bar4[2],bar5[0],bar5[1],bar5[2]]
plt.bar(bar1,h,w,label="Healthy")
plt.bar(bar2,s1,w,label="Sick1")
plt.bar(bar3,s2,w,label="Sick2")
plt.bar(bar4,s3,w,label="Sick3")
plt.bar(bar5,s4,w,label="Sick4")
label = [cat1[0],cat2[0],cat3[0],cat1[1],cat2[1],cat3[1],cat1[2],cat2[2],cat3[2],cat1[3],cat2[3],cat3[3],cat1[4],cat2[4],cat3[4]]

plt.xlabel("Categories of Blood Pressure")
plt.ylabel("Total Quantity")
plt.title("Visualizing dataset for each BP category")
plt.xticks(bar2,x)
# Text on the top of each barplot
for i in range(len(bart)):
    plt.text(x = bart[i]-0.05 , y = st[i], s = label[i], size = 10)
plt.legend()
plt.legend()

plt.show()
#



