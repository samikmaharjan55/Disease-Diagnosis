#================================================================================================================
#TITLE: DETERMINING MEMBERSHIP FUNCTION FOR FASTING BLOOD SUGAR
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
#DEFINING INPUT FOR BLOOD SUGAR FOR DATA PLOTTING.
#CREATING NPARRAY OF OF ZEROS SIMILAR TO SIZE INPUT FOR REPLACING LATER ON.
bsugar = np.linspace(0,1.5,400)
high_bsugar1 = np.zeros_like(bsugar)
low_bsugar1 = np.zeros_like(bsugar)
# bsugar2 = np.zeros_like(bsugar)


#================================================================================================================
#GENERATING MEMBERSHIP INPUT VALUES FOR BLOOD SUGAR USING SELECTED MEMBERSHIP
for i in range(len(bsugar)):
    #INPUT TYPE 1 INCREASING FUNCTION
    high_bsugar1[i]=mm.flat(bsugar[i],1)
    low_bsugar1[i]= mm.revflat(bsugar[i],1)

    #INPUT TYPE 2. SIGMOID INCREASING
    # bsugar2[i] = mm.sigmoid(bsugar[i], 0.45, 106)


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
#EVALUATING BLOOD SUGAR LEVELS FOR INPUT DATA TAKEN FROM RAW DATA.
input_bsugar = dfnew.fbs.to_numpy().astype(int).tolist() #BLOOD SUGAR DATA CONVERTED INTO LIST VARIABLE
target = dfnew.target.to_numpy().astype(int).tolist() #TARGET VALUE CONVERTED INTO LIST VARIABLE.
#EMPTY LIST TO BE APPENDED TO LATER ON (1 2 AND 3 CORRESPOND TO THE DIFFERENT COMBINATIONS OF MEMBERSHIP FUCNTIONS AS EXPLAINED ABOVE)
in_high_bsugar1 = []
in_low_bsugar1 = []


#Filtering dataset based on ranges for each Fuzzy class
def segbs (arr1,arr2,val1,val2):
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

cat1 = segbs(input_bsugar,target,0,0.99999)
cat2 = segbs(input_bsugar,target,1,1.5)


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
#CREATING AN EMPTY OUTPUT ARRAY TO BE APPENDED TO LATER.
output_1 = []
output_2 = []

##================================================================================================================
#GENERATING RULES
#Rules
#low blood sugar --> Healthy (0)
#high blood sugar --> SICK 1 OR SICK 2 OR SICK 3 OR SICK 4

#EVALUATING RULE VALUES BASED ON INPUT PARAMETERS FOR MEMBERSIP FUCNTION COMBINDATIONS 1 AND 2
for i in range(len(input_bsugar)):
    #R11 - RULE 1 FOR FUNCTION 1
    R11 = np.fmin(in_low_bsugar1[i],healthy)


    #R21 - RULE 2 FOR FUNCTION 1
    sick1234 = np.maximum(sick1, np.maximum(sick2, np.maximum(sick3, sick4)))
    R21 = np.fmin(in_high_bsugar1[i],sick1234)

    #AGGREGATION OF RULES
    R1 = np.maximum(R11,R21)

    #DEFUZZIFICATION
    output1 = np.trapz(R1 * out_test, out_test) / np.trapz(R1,out_test)  # calculating centroid for union of rules.
    if np.isnan(output1):
        output1 = 0

        #FOR TESTING PURPOSES
        # # Simple rules from membership definition
        # R_bsugar1 = np.fmin(in_low_bsugar1[i], low_risk)
        # R_bsugar2 = np.fmin(in_high_bsugar1[i], np.maximum(med_risk, high_risk))
        # R_bsugar = np.maximum(R_bsugar2, R_bsugar1)
        # outputbsugar = np.trapz(R_bsugar * out_test, out_test) / np.trapz(R_bsugar,out_test)  # calculating centroid for union of rules.
        # if np.isnan(outputbsugar):
        #     outputbsugar = 0
        # output_bsugar.append(outputbsugar)

    #APPEND TO THE RESPECTIVE OUTPUT LIST
    output_1.append(output1)


##================================================================================================================
#EVALUATING CONFUSION MATRIX AND ACCURACY
for i in range(len(output_1)):
    # print(output_1[i])
    if output_1[i] < 0.5:
        output_1[i] = 0
    elif (output_1[i] >=0.5) and (output_1[i] <1.5):
         output_1[i] = 1
    elif (output_1[i]>= 1.5) and (output_1[i] <2.5):
         output_1[i] = 2
    elif (output_1[i] >= 2.5) or (output_1[i] < 3.5):
        output_1[i] = 3
    elif output_1[i]>= 3.5:
         output_1[i] = 4

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

cf1 = confusion_matrix(target,output_1)
acc1 = accuracy_score(target,output_1)


print("accuracy output 1:",acc1*100,"%")

##================================================================================================================
#PLOTTING DATA
plt.figure("Model")
plt.subplot(2,3,1)
plt.title("Blood Sugar Linear")
plt.plot(bsugar,high_bsugar1,label="High Blood Sugar")
plt.plot(bsugar,low_bsugar1,label="Low Blood Sugar")
plt.xlabel("Blood Sugar")
plt.ylabel("Fuzzy output")
plt.legend()


plt.subplot(2,3,2)
plt.title("Output Test")
plt.plot(out_test,healthy,label="Healthy")
plt.plot(out_test,sick1,label="Sick1")
plt.plot(out_test,sick2,label="Sick2")
plt.plot(out_test,sick3,label="Sick3")
plt.plot(out_test,sick4,label="sick4")
plt.xlabel("Category of output")
plt.ylabel("Fuzzy output")
plt.legend()


plt.subplot(2,3,3)
w=0.1
x = ["Low Blood Sugar","High Blood Sugar"]
h = [cat1[0],cat2[0]]
s1 = [cat1[1],cat2[1]]
s2 = [cat1[2],cat2[2]]
s3 = [cat1[3],cat2[3]]
s4 = [cat1[4],cat2[4]]
st = h+s1+s2+s3+s4

bar1 = np.arange(len(x))
bar2 = [i+w for i in bar1]
bar3 = [i+2*w for i in bar1]
bar4 = [i+3*w for i in bar1]
bar5 = [i+4*w for i in bar1]
bart = [bar1[0],bar1[1],bar2[0],bar2[1],bar3[0],bar3[1],bar4[0],bar4[1],bar5[0],bar5[1]]
plt.bar(bar1,h,w,label="Healthy")
plt.bar(bar2,s1,w,label="Sick1")
plt.bar(bar3,s2,w,label="Sick2")
plt.bar(bar4,s3,w,label="Sick3")
plt.bar(bar5,s4,w,label="Sick4")

label = [cat1[0],cat2[0],cat1[1],cat2[1],cat1[2],cat2[2],cat1[3],cat2[3],cat1[4],cat2[4]]

plt.xlabel("Categories of Blood Sugar")
plt.ylabel("Total Quantity")
plt.title("Visualizing dataset for each Blood Sugar category")
plt.xticks(bar2,x)
# Text on the top of each barplot
for i in range(len(bart)):
    plt.text(x = bart[i]-0.05 , y = st[i], s = label[i], size = 10)
plt.legend()

plt.show()




