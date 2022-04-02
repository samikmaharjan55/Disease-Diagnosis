#================================================================================================================
#TITLE: DETERMINING MEMBERSHIP FUNCTION FOR CHEST PAIN
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

#================================================================================================================
#MAMDANI METHOD
#DEFINING INPUT FOR CHEST PAIN FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
cpv = np.linspace(0,5,800)
cp11 = np.zeros_like(cpv)
cp21 = np.zeros_like(cpv)
cp31 = np.zeros_like(cpv)
cp41 = np.zeros_like(cpv)

#================================================================================================================
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

# #================================================================================================================
#EVALUATING CHEST PAIN LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_cp = dfnew.cp.to_numpy().astype(int).tolist() #CHOLESTEROL DATA CONVERTED INTO LIST VARIABLE
target = dfnew.target.to_numpy().astype(int).tolist() #TARGET VALUE CONVERTED INTO LIST VARIABLE.

#Filtering dataset based on ranges for each Fuzzy class
def segcp (arr1,arr2,cpval):
    o0 = 0
    o1 = 0
    o2 = 0
    o3 = 0
    o4 = 0
    for i in range(len(arr1)):
        if arr1[i] == cpval:
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

cat1 = segcp(input_cp,target,1)
cat2 = segcp(input_cp,target,2)
cat3 = segcp(input_cp,target,3)
cat4 = segcp(input_cp,target,4)
# print(cat1)

#EMPTY LIST TO BE APPENDED TO LATER ON (1 2 AND 3 CORRESPOND TO THE DIFFERENT COMBINATIONS OF MEMBERSHIP FUCNTIONS AS EXPLAINED ABOVE)
cpi1 = []
cpi2 = []
cpi3 = []
cpi4 = []

# #EVALUATING CHOLESTEROL LEVELS USING MEMBERSHIP FUNCTIONS SELECTED
for i in range(len(input_cp)):
    cpi1.append(mm.tri(input_cp[i], 0.5, 1, 1.5))
    cpi2.append(mm.tri(input_cp[i], 1.5, 2, 2.5))
    cpi3.append(mm.tri(input_cp[i], 2.5, 3, 3.5))
    cpi4.append(mm.tri(input_cp[i], 3.5, 4, 4.5))
#     # print(input_bp[i]," ",in_low_bp_1[i]," ",in_med_bp_1[i], " ",in_high_bp_1[i] ) #FOR TESTING AND CHECKING PURPOSES

cpi1 = cp_round(cpi1)
cpi2 = cp_round(cpi2)
cpi3 = cp_round(cpi3)
cpi4 = cp_round(cpi4)

#================================================================================================================
#CREATING AN EMPTY OUTPUT ARRAY TO BE APPENDED TO LATER.
output_1 = []

##================================================================================================================
#GENERATING RULES
#Rules
case1 = np.maximum(healthy,sick1)
case2 = np.maximum(sick1,sick2)
case3 = np.maximum(sick3,sick4)
#
# #EVALUATING RULE VALUES BASED ON INPUT PARAMETERS FOR MEMBERSHIP FUNCTION
for i in range(len(input_cp)):
    #RULE 1 FOR FUNCTION 1
    R1 = np.fmin(cpi4[i],case1)

    #RULE 2 FOR FUNCTION 1
    R2 = np.fmin(cpi3[i],np.maximum(healthy,case2))

    #RULE 3 FOR FUNCTION 1
    R3 = np.fmin(cpi2[i],case3)

    #RULE 4 FOR FUNCTION 1
    R4 = np.fmin(cpi1[i],case3)

    #UNION OF RULES
    R_1 = np.maximum(R4,np.maximum(R3,np.maximum(R2,R1)))
    output1 = np.trapz(R_1 * out_test, out_test) / np.trapz(R_1,out_test)  # calculating centroid for union of rules.
    #TO CHECK IF THERE'S AN ERROR
    if np.isnan(output1):
        output1 = 0
    #APPEND TO THE RESPECTIVE OUTPUT LIST
    output_1.append(output1)

#EVALUATING CONFUSION MATRIX AND ACCURACY
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
tround = roundup(target)
outround1 = roundup(output_1)


cf1 = confusion_matrix(target,output_1)
acc1 = accuracy_score(target,output_1)

print("accuracy output 1:",acc1*100,"%")

##================================================================================================================

#PLOTTING DATA
plt.figure("Chest Pain Membership Function Selection")
plt.subplot(2,3,1)
plt.title("Chest Pain Triangle Membership Function")
plt.xlabel("Chest Pain")
plt.ylabel("Fuzzy output")
plt.plot(cpv,cp11,label="Chest Pain 1")
plt.plot(cpv,cp21,label="Chest Pain 2")
plt.plot(cpv,cp31,label="Chest Pain 3")
plt.plot(cpv,cp41,label="Chest Pain 4")
plt.legend()
#
plt.subplot(2,3,2)
plt.title("Triangle Function Adjusted")
plt.xlabel("Chest Pain")
plt.ylabel("Fuzzy output")
plt.plot(cpv,cp1r,label="Chest Pain 1")
plt.plot(cpv,cp2r,label="Chest Pain 2")
plt.plot(cpv,cp3r,label="Chest Pain 3")
plt.plot(cpv,cp4r,label="Chest Pain 4")
plt.legend()

plt.subplot(2,3,3)
w=0.1
x = [1,2,3,4]
h = [cat1[0],cat2[0],cat3[0],cat4[0]]
s1 = [cat1[1],cat2[1],cat3[1],cat4[1]]
s2 = [cat1[2],cat2[2],cat3[2],cat4[2]]
s3 = [cat1[3],cat2[3],cat3[3],cat4[3]]
s4 = [cat1[4],cat2[4],cat3[4],cat4[4]]
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
# print(bar1)
bart = [bar1[0],bar1[1],bar1[2],bar1[3],bar2[0],bar2[1],bar2[2],bar2[3],bar3[0],bar3[1],bar3[2],bar3[3],bar4[0],bar4[1],bar4[2],bar4[3],bar5[0],bar5[1],bar5[2],bar5[3]]
plt.bar(bar1,h,w,label="Healthy")
plt.bar(bar2,s1,w,label="Sick1")
plt.bar(bar3,s2,w,label="Sick2")
plt.bar(bar4,s3,w,label="Sick3")
plt.bar(bar5,s4,w,label="Sick4")
label = [cat1[0],cat2[0],cat3[0],cat4[0],cat1[1],cat2[1],cat3[1],cat4[1],cat1[2],cat2[2],cat3[2],cat4[2],cat1[3],cat2[3],cat3[3],cat4[3],cat1[4],cat2[4],cat3[4],cat4[4]]

plt.xlabel("Categories of Chest Pain (1,2,3,4)")
plt.ylabel("Total Quantity")
plt.title("Visualizing dataset for each CP category")
plt.xticks(bar3,x)
# Text on the top of each barplot
for i in range(len(bart)):
    plt.text(x = bart[i]-0.05 , y = st[i], s = label[i], size = 8)
plt.legend()

plt.show()
# #
#
#

