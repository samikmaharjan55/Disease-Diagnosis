#================================================================================================================
#TITLE: DETERMINING MEMBERSHIP FUNCTION FOR AGE
#Written by: LOH KHAI REN
#TPNUM: TP062775
#Date: 16-Feb-2021
#================================================================================================================
#IMPORTING NECESSARY PYTHON PACKAGES
import pandas as pd
import Memberships as mm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
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
# print(dfnew)
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
#DEFINING INPUT FOR AGE FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
age = np.linspace(0,100,1200)
young_age1 = np.zeros_like(age)
mid_age1 = np.zeros_like(age)
old_age1 = np.zeros_like(age)

young_age2 = np.zeros_like(age)
mid_age2 = np.zeros_like(age)
old_age2 = np.zeros_like(age)

young_age3 = np.zeros_like(age)
mid_age3 = np.zeros_like(age)
old_age3 = np.zeros_like(age)

#================================================================================================================
#GENERATING MEMBERSHIP INPUT VALUES FOR AGE USING SELECTED MEMBERSHIP
for i in range(len(age)):
    #INPUT TYPE 1 DECREASING, TRIANGLE AND INCREASING FUNCTION
    young_age1[i] = mm.dec(age[i], 27, 33)
    mid_age1[i] = mm.tri(age[i], 32, 39, 46)
    old_age1[i] = mm.inc(age[i], 45, 51)

    #INPUT TYPE 2. SIGMOID DECREASING, GAUSSIAN AND SIGMOID INCREASING
    young_age2[i] = mm.sigmoid(age[i], -3, 30)
    mid_age2[i] = mm.gaus(age[i], 39, 2)
    old_age2[i] = mm.sigmoid(age[i], 3, 48)

    #INPUT TYPE 3, DECREASING, TRIANGLE AND INCREASING FUNCTION (WITH EXCEPTION)
    young_age3[i]=mm.dec(age[i],27,33)
    mid_age3[i]=mm.tri(age[i],33,39,44)
    old_age3[i] = mm.inc(age[i],44,51)

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
#EVALUATING AGE LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_age = dfnew.age.to_numpy().astype(int).tolist() #AGE DATA CONVERTED INTO LIST VARIABLE
target = dfnew.target.to_numpy().astype(int).tolist() #TARGET VALUE CONVERTED INTO LIST VARIABLE.

#EMPTY LIST TO BE APPENDED TO LATER ON (1 2 AND 3 CORRESPOND TO THE DIFFERENT COMBINATIONS OF MEMBERSHIP FUCNTIONS AS EXPLAINED ABOVE)
in_young_age1 = []
in_mid_age1 = []
in_old_age1 = []
#
in_young_age2 = []
in_mid_age2 = []
in_old_age2 = []
#
in_young_age3 = []
in_mid_age3 = []
in_old_age3 = []

#Filtering dataset based on ranges for each Fuzzy class
def segage (arr1,arr2,val1,val2):
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

cat1 = segage(input_age,target,0,33)
cat2 = segage(input_age,target,33,44)
cat3 = segage(input_age,target,44,100)

#EVALUATING AGE LEVELS USING MEMBERSHIP FUNCTIONS SELECTED
for i in range(len(input_age)):

    in_young_age1.append(mm.dec(input_age[i], 27, 33))
    in_mid_age1.append(mm.tri(input_age[i], 32, 39, 46))
    in_old_age1.append(mm.inc(input_age[i], 45, 51))

    in_young_age2.append(mm.sigmoid(input_age[i], -3, 30))
    in_mid_age2.append(mm.gaus(input_age[i], 39, 2))
    in_old_age2.append(mm.sigmoid(input_age[i], 3, 48))

    in_young_age3.append(mm.dec(input_age[i],27,33))
    in_mid_age3.append(mm.tri(input_age[i],33,39,44))
    in_old_age3.append(mm.inc(input_age[i],44,51))

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
#CREATING AN EMPTY OUTPUT ARRAY TO BE APPENDED TO LATER.
output_1 = []
output_2 = []
output_3 = []

##================================================================================================================
#GENERATING RULES
#Rules
#young age --> Healthy (0)
#mid age --> SICK 1 or SICK 2
#old age --> SICK 3 or SICK 4
pos = []
#EVALUATING RULE VALUES BASED ON INPUT PARAMETERS FOR MEMBERSIP FUCNTION COMBINDATIONS 1 2 AND 3
for i in range(len(input_age)):
    #R11 - RULE 1 FOR FUNCTION 1
    R11 = np.fmin(in_young_age1[i],healthy)

    #R21 - RULE 2 FOR FUNCTION 1
    sick12 = np.maximum(sick1, sick2)
    R21 = np.fmin(in_mid_age1[i],sick12)


    #RULE 31 - RULE 3 FOR FUNCTION 1
    sick34 = np.maximum(sick3, sick4)
    R31 = np.fmin(in_old_age1[i],sick34)

    #UNION OF RULES
    R_1 = np.maximum(R31,np.maximum(R21,R11))
    output1 = np.trapz(R_1 * out_test, out_test) / np.trapz(R_1,out_test)  # calculating centroid for union of rules.
    #TO CHECK IF THERE'S AN ERROR
    if np.isnan(output1):
        output1 = 0
        print("From output1 ",in_young_age1[i]," ",in_mid_age1[i]," ",in_old_age1[i])
    #APPEND TO THE RESPECTIVE OUTPUT LIST
    output_1.append(output1)

    #FUNCTION 2
    R12 = np.fmin(in_young_age2[i],healthy)
    sick12 = np.maximum(sick1, sick2)
    R22 = np.fmin(in_mid_age2[i],sick12)
    sick34 = np.maximum(sick3, sick4)
    R32 = np.fmin(in_old_age2[i], sick34)
    R_2 = np.maximum(R32, np.maximum(R22, R12))
    output2 = np.trapz(R_2 * out_test, out_test) / np.trapz(R_2, out_test)  # calculating centroid for union of rules.
    if np.isnan(output2):
        output2 = 0
        print("From output2 ",in_young_age2[i]," ",in_mid_age2[i]," ",in_old_age2[i])
    output_2.append(output2)

    #FUNCTION 3
    R13 = np.fmin(in_young_age3[i], healthy)
    sick12 = np.maximum(sick1, sick2)
    R23 = np.fmin(in_mid_age3[i], sick12)
    sick34 = np.maximum(sick3, sick4)
    R33 = np.fmin(in_old_age3[i], sick34)
    R_3 = np.maximum(R33, np.maximum(R23, R13))
    output3 = np.trapz(R_3 * out_test, out_test) / np.trapz(R_3, out_test)  # calculating centroid for union of rules.
    if np.isnan(output3):
        output3 = 0
        print("From output3 ",in_young_age3[i]," ",in_mid_age3[i]," ",in_old_age3[i])
    output_3.append(output3)



##================================================================================================================
#EVALUATING CONFUSION MATRIX AND ACCURACY
for i in range(len(output_1)):
    if output_1[i] < 0.5:
        output_1[i] = 0
    elif (output_1[i] >=0.5) or (output_1[i] <=1.5):
         output_1[i] = 1
    elif (output_1[i]>= 1.5) or (output_1[i] <=2.5):
         output_1[i] = 2
    elif (output_1[i] >= 2.5) or (output_1[i] <= 3.5):
        output_1[i] = 3
    elif output_1[i]> 3.5:
         output_1[i] = 4


for i in range(len(output_2)):
    # print(output_2[i])
    if output_2[i] < 0.5:
        output_2[i] = 0
    elif (output_2[i] >=0.5) or (output_2[i] <=1.5):
         output_2[i] = 1
    elif (output_2[i]>= 1.5) or (output_2[i] <=2.5):
         output_2[i] = 2
    elif (output_2[i] >= 2.5) or (output_2[i] <= 3.5):
        output_2[i] = 3
    elif output_2[i]> 3.5:
         output_2[i] = 4
    # print(output_2[i])
# print(output_2)

for i in range(len(output_3)):
    # print(output_3[i])
    if output_3[i] < 0.5:
        output_3[i] = 0
    elif (output_3[i] >=0.5) or (output_3[i] <=1.5):
         output_3[i] = 1
    elif (output_3[i]>= 1.5) or (output_3[i] <=2.5):
         output_3[i] = 2
    elif (output_3[i] >= 2.5) or (output_3[i] <= 3.5):
        output_3[i] = 3
    elif output_3[i]> 3.5:
         output_3[i] = 4
    # print(output_3[i])
# print(output_3)

def roundup(arr):
    for i in range(len(arr)):
        # print(output3[i]
        if arr[i] < 2.5:
            arr[i] = 0
        elif arr[i]>= 2.5 and arr[i]< 3.5:
            arr[i] = 1
        elif arr[i] >= 3.5:
            arr[i] = 2
    return arr

tround = roundup(target)
outround1 = roundup(output_1)
outround2 = roundup(output_2)
outround3 = roundup(output_3)


cf1 = confusion_matrix(target,output_1)
acc1 = accuracy_score(target,outround1)

cf2 = confusion_matrix(target,output_2)
acc2 = accuracy_score(target,outround2)

cf3 = confusion_matrix(target,output_3)
acc3 = accuracy_score(target,outround3)

print("accuracy output 1:",acc1*100,"accuracy output 2:",acc2*100,"accuracy output 3:",acc3*100)

##================================================================================================================
#PLOTTING DATA
plt.figure("Model")
plt.subplot(2,3,1)
plt.title("Age Linear Class Overlapping - Set 1")
plt.plot(age,young_age1,label="Young")
plt.plot(age,mid_age1,label="Mid")
plt.plot(age,old_age1,label="Old")
plt.xlabel("Age (Years)")
plt.ylabel("Fuzzy output")
plt.legend()

plt.subplot(2,3,2)
plt.title("Age Sigmoid - Set 2")
plt.plot(age,young_age2,label="Young")
plt.plot(age,mid_age2,label="Mid")
plt.plot(age,old_age2,label="Old")
plt.xlabel("Age (Years)")
plt.ylabel("Fuzzy output")
plt.legend()

plt.subplot(2,3,3)
plt.title("Age Linear No Class Overlapping - Set 3")
plt.plot(age,young_age3,label="Young")
plt.plot(age,mid_age3,label="Mid")
plt.plot(age,old_age3,label="Old")
plt.xlabel("Age (Years)")
plt.ylabel("Fuzzy output")
plt.legend()

plt.subplot(2,3,4)
plt.title("Output Test")
plt.plot(out_test,healthy,label="Healthy")
plt.plot(out_test,sick1,label="Sick1")
plt.plot(out_test,sick2,label="Sick2")
plt.plot(out_test,sick3,label="Sick3")
plt.plot(out_test,sick4,label="sick4")
plt.xlabel("Category of output")
plt.ylabel("Fuzzy output")
plt.legend()

plt.subplot(2,3,5)
w=0.1
x = ["Young","Medium","Old"]
h = [cat1[0],cat2[0],cat3[0]]
s1 = [cat1[1],cat2[1],cat3[1]]
s2 = [cat1[2],cat2[2],cat3[2]]
s3 = [cat1[3],cat2[3],cat3[3]]
s4 = [cat1[4],cat2[4],cat3[4]]
st = h+s1+s2+s3+s4

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

plt.xlabel("Categories of Age")
plt.ylabel("Total Quantity")
plt.title("Visualizing dataset for each Age category")
plt.xticks(bar2,x)
# Text on the top of each barplot
for i in range(len(bart)):
    plt.text(x = bart[i]-0.05 , y = st[i], s = label[i], size = 10)
plt.legend()

plt.show()



