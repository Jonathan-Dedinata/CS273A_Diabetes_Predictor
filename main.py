import mltools as ml
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import graphviz

class patient:

    # patient_nbr = None
    # gender = None
    # age = None
    # admission_type_id = None
    # discharge_type = None
    # admission_source = None
    # hosp_duration = None
    # num_lab_procedures = None
    # num_medications = None
    # insulin_status = None
    # medicated = None
    # readmitted = None


    def __init__(self, patient_nbr, gender, age, admission_type, discharge_type, admission_source, hosp_duration, num_lab_procedures, num_procedures, num_medications, emergencies, insulin_status, medicated, readmitted):
        self.patient_nbr = patient_nbr
        self.gender = gender
        self.age = age
        self.admission_type_id = admission_type
        self.discharge_type = discharge_type
        self.admission_source = admission_source
        self.hosp_duration = hosp_duration
        self.num_lab_procedures = num_lab_procedures
        self.num_procedures = num_procedures
        self.num_medications = num_medications
        self.emergencies = emergencies
        self.insulin_status = insulin_status
        self.medicated = medicated
        self.readmitted = readmitted


def main():
    dataset = np.genfromtxt('dataset_diabetes/diabetic_data.csv', delimiter= ',', dtype = None, encoding = None)
    # print(dataset.shape)
    patients = []
    # for features to dissect (starts at 0): 1, 3, 4, 6, 7, 8, 9, 12, 13, 14, 16, 41, 48, 49
    # for i in range(1, 101767):
    #     pnbr = dataset[i,1]
    #     gen = dataset[i,3]
    #     age = dataset[i,4]
    #     admtype = dataset[i,6]
    #     distype = dataset[i,7]
    #     adm_source = dataset[i,8]
    #     hosp_dur = dataset[i,9]
    #     numlabproc = dataset[i, 12]
    #     numproc = dataset[i, 13]
    #     num_meds = dataset[i, 14]
    #     emergencies = dataset[i,16]
    #     ins_status = dataset[i, 41]
    #     medicated = dataset[i, 48]
    #     readmitted = dataset[i, 49]
    #     new_patient = patient(pnbr,gen,age,admtype,distype,adm_source,hosp_dur,numlabproc,numproc, num_meds, ins_status, medicated, readmitted)
    #     patients.append(new_patient)
    #ig_rankings = information_gain(dataset)
    indexes = [4,6,7,8,9,12,13,14,16,41,48,49]
    cleaved_dataset = dataset[:, indexes]
    binary_dataset = converttobinary(cleaved_dataset)
    createDecisionTree(binary_dataset)


def converttobinary(dataset):
    INDEX = 101767
    binary_dataset = np.zeros((INDEX,12))

    for i in range(1, INDEX):
        for j in range(12):
            if i == 0:
                binary_dataset[i][j] = dataset[i][j]
            else:
                ## age
                if j == 0:
                    age = dataset[i][0]
                    if age == '[0-10)' or '[10-20)'or '[20-30' or '[30-40':
                        binary_dataset[i][0] = 0
                    else: binary_dataset[i][0] = 1
                ## admission type
                elif j == 1:
                    admtype = int(dataset[i][1])
                    if admtype == 3 or admtype == 4 or admtype == 5 or admtype == 6 or admtype == 8:
                        binary_dataset[i][1] = 0
                    else: binary_dataset[i][1] = 0
                ## number inpatients

                ## discharge type
                elif j == 2:
                    distype = int(dataset[i][2])
                    if distype == 1 or distype == 2  or distype == 6 or distype == 8 or distype == 11 or distype == 18 or distype == 19 or distype == 20 or distype == 21 or distype == 25 or distype == 26:
                        binary_dataset[i][2] = 0
                    else: binary_dataset[i][2] = 1
                ## admission source id
                elif j == 3:
                    admsrc = int(dataset[i][[3]])
                    if admsrc == 11 or admsrc == 12 or admsrc == 13 or admsrc == 14 or admsrc == 21 or admsrc == 23 or admsrc == 24:
                        binary_dataset[i][3] = 0
                    else: binary_dataset[i][3] = 1
                ## hospital duration
                elif j == 4:
                    hospdur = int(dataset[i][4])
                    if hospdur <= 7:
                        binary_dataset[i][4] = 0
                    else: binary_dataset[i][4] = 1
                elif j == 5:
                    numlp = int(dataset[i][j])
                    if numlp <= 60:
                        binary_dataset[i][j] = 0
                    else: binary_dataset[i][j] = 1
                elif j== 6:
                    numproc = int(dataset[i][j])
                    if numproc > 0:
                        binary_dataset[i][j]  = 0
                    else: binary_dataset[i][j] = 1
                elif j == 7:
                    nummeds = int(dataset[i][j])
                    if nummeds < 40:
                        binary_dataset[i][j] = 0
                    else: binary_dataset[i][j] = 1
                elif j == 8:
                    emergencies = int(dataset[i][j])
                    if emergencies > 0:
                        binary_dataset[i][j] = 0
                    else: binary_dataset[i][j] = 1
                elif j == 9:
                    insulin = dataset[i,j].upper()
                    if insulin == 'Down' or insulin == 'Up':
                        binary_dataset[i][j] = 0
                    else: binary_dataset[i][j] = 1
                elif j == 10:
                    medicated = dataset[i,j].upper()
                    if medicated == 'NO':
                        binary_dataset[i][j] = 0
                    else: binary_dataset[i][j] = 1
                elif j == 11:
                    readmitted = dataset[i,j].upper()
                    if readmitted == 'NO':
                        binary_dataset[i][j] = 0
                    else: binary_dataset[i][j] = 1
    return binary_dataset





def information_gain(dataset):
    readmitted_counter = 0
    ig_rankings = []
    INDEX = 101767
    ## gender
    male_count = 0
    female_count = 0
    male_readmitted = 0
    female_readmitted = 0

    ## age
    age_arr = np.zeros((10,2))

    ## admission type
    admtype_arr = np.zeros((8,2))
    a = np.zeros((2,2))

    ## discharge type
    dis_arry = np.zeros((30,2))

    ## admission source
    admsrc_arr = np.zeros((26,2))

    ## hospoital duration
    hospdur_arr = np.zeros((14,2))

    ## number of lab procedures
    numlp_arr = np.zeros((14,2))

    ## number of procedures
    numproc_arr = np.zeros((7,2))

    ## number of medications
    nummeds_arr = np.zeros((9,2))

    ## number of emergencies
    emergencies_arr = np.zeros((2,2))
    em_arr = np.zeros((8,2))

    ## insulin levels
    insulin_arr = np.zeros((4,2 ))

    ## taking diabetes meds
    medicated_arr = np.zeros((2,2))

    for i in range(1, INDEX):
        ## gender
        readmitted = dataset[i][49].upper() != "NO"
        if dataset[i][3] == 'Male':
            male_count += 1
            if readmitted:
                male_readmitted+=1
        elif dataset[i][3] == 'Female':
            female_count += 1
            if readmitted:
                female_readmitted+=1

        ## age
        age_range = dataset[i][4]
        age = int(age_range[1:2:1])
        age_arr[age][0] +=1
        if readmitted:
            age_arr[age][1] += 1


        ##adm type


        admtype = int(dataset[i][6]) -1
        adt = int(dataset[i][6])
        if adt == 3 or adt == 4 or adt == 5 or adt == 6 or adt == 8:
            a[0][0]+=1
            if readmitted: a[0][1] += 1
        else:
            a[1][0]+=1
            if readmitted: a[1][1] +=1

        admtype_arr[admtype][0] +=1
        if readmitted:
            admtype_arr[admtype][1] +=1

        ## discharge type
        distype = int(dataset[i][7]) - 1
        dis_arry[distype][0] +=1
        if readmitted:
            dis_arry[distype][1] +=1

        ## adm source
        admsrc = int(dataset[i][8]) -1
        admsrc_arr[admsrc][0] +=1
        if readmitted:
            admsrc_arr[admsrc][1] +=1

        ## hospital duration
        hospdur = int(dataset[i][9]) -1
        hospdur_arr[hospdur][0] +=1
        if readmitted:
            hospdur_arr[hospdur][1] +=1

        ## number of lab procedures
        numlp = int(int(dataset[i][12]) / 10)
        numlp_arr[numlp][0] +=1
        if readmitted:
            numlp_arr[numlp][1] +=1

        ## number of procedures
        numproc = int(dataset[i][13])
        numproc_arr[numproc][0] +=1
        if readmitted:
            numproc_arr[numproc][1] +=1

        ## number of medications
        nummeds = int(int(dataset[i][14]) / 10)
        nummeds_arr[nummeds][0] +=1
        if readmitted:
            nummeds_arr[nummeds][1] +=1

        ## number of emergencies
        emergency = int(dataset[i][16])
        em = int(emergency/10)
        em_arr[em][0] += 1
        if readmitted: em_arr[em][1] += 1
        if emergency > 0 :
            emergencies_arr[1][0] +=1
            emergency = 1
        else:
            emergencies_arr[0][0] +=1
            emergency = 0
        if readmitted:
            emergencies_arr[emergency][1] +=1

        ## insulin levels
        insulin = dataset[i][41].upper()
        if insulin == 'UP':
            insulin_arr[0][0] +=1
            if readmitted: insulin_arr[0][1] +=1
        elif insulin == 'DOWN':
            insulin_arr[1][0] +=1
            if readmitted: insulin_arr[1][1] +=1
        elif insulin == 'STEADY':
            insulin_arr[2][0] +=1
            if readmitted: insulin_arr[2][1] +=1
        elif insulin == 'NO':
            insulin_arr[3][0] +=1
            if readmitted: insulin_arr[3][1] +=1

        # taking diabetes medications
        medicated = dataset[i][48].upper()
        if medicated == 'YES':
            medicated_arr[0][0] +=1
            if readmitted: medicated_arr[0][1] +=1
        elif medicated == 'NO':
            medicated_arr[1][0] +=1
            if readmitted: medicated_arr[1][1] +=1
        else: print(medicated)
        if readmitted: readmitted_counter+=1
    ## gender
    readmitted_percent = (male_readmitted + female_readmitted)/ INDEX
    print(readmitted_percent)
    readmit_e = entropy(readmitted_percent)
    readm_tot = male_readmitted + female_readmitted
    male_readmitted_percent = male_readmitted/ male_count
    female_readmitted_percent = female_readmitted/ female_count
    ig_gender = readmit_e - ((male_count/INDEX)*entropy(male_readmitted_percent) + (female_count/INDEX)*entropy(female_readmitted_percent))
    #print('The information gain found from the gender feature is:',ig_gender)
    ig_rankings.append((ig_gender, 'gender'))

    ## age
    age_rankings = []
    age_e_sum = 0.0
    for i in range(10):
        if age_arr[i][0] != 0:
            age_rankings.append(age_arr[i][1]/age_arr[i][0])
            age_e_sum += ((age_arr[i][0]/INDEX)*entropy(age_arr[i][1]/age_arr[i][0]))
    ig_age = readmit_e - age_e_sum
    #print('The information gain found from the age range feature is:',ig_age )
    barplot(range(10),age_rankings,'Decade of Age (I.E. 9 = [90-100))',False)
    #age_rankings.sort(key=lambda i: i[0], reverse=False)
    # print(age_rankings)
    ig_rankings.append((ig_age, 'age'))

    ## admission type
    aind = a[0][0]/INDEX
    e1 = entropy(a[0][1]/a[0][0])
    e2 = entropy(a[1][1]/a[1][0])
    ig_at  = readmit_e - ((a[0][0]/INDEX) * entropy(a[0][1]/a[0][0])) + ((a[1][0]/INDEX)*entropy(a[1][1]/a[1,0]))
    #print('The information gain fround from the fixed at feat is:' , ig_at)

    admtype_e_sum = 0.0
    admtype_rankings = []
    for i in range(8):
        if admtype_arr[i][0] != 0:
            admtype_rankings.append(admtype_arr[i][1]/admtype_arr[i][0])
            admtype_e_sum  += ((admtype_arr[i][0]/INDEX)*entropy(admtype_arr[i][1]/admtype_arr[i][0]))
        else: admtype_rankings.append(0)
    ig_admtype = readmit_e - admtype_e_sum
    #print('The information gain found from the admission type feature is:',ig_admtype)
    ig_rankings.append((ig_admtype, 'admission type'))
    #print(admtype_rankings)
    barplot(np.arange(1,9), admtype_rankings, 'Admission Type ID', True)
    # admtype_rankings.sort(key=lambda i: i[0], reverse=False)
    #print(admtype_rankings)

    ## discharge type
    dis_e_sum = 0.0
    distype_rankings = []
    for i in range(30):
        if dis_arry[i][0] != 0:
            distype_rankings.append(dis_arry[i][1] / dis_arry[i][0])
            dis_e_sum += ((dis_arry[i][0]/INDEX)*entropy(dis_arry[i][1]/dis_arry[i][0]))
        else: distype_rankings.append(0)
    ig_distype = readmit_e - dis_e_sum
    #print('The information gain found from the discharge type feature is:',ig_distype)
    barplot(np.arange(1,31), distype_rankings, 'Discharge Type ID', True)
    ig_rankings.append((ig_distype, 'discharge type'))

    ## admission source
    admsrc_e_sum = 0.0
    admsrc_rankings = []
    for i in range(26):
        if admsrc_arr[i][0] != 0:
            admsrc_rankings.append(admsrc_arr[i][1]/admsrc_arr[i][0])
            admsrc_e_sum += ((admsrc_arr[i][0]/INDEX) * entropy(admsrc_arr[i][1]/admsrc_arr[i][0]))
        else: admsrc_rankings.append(0)
    ig_admsrc = readmit_e - admsrc_e_sum
    #print('The information gain found from the admission source feature is:',ig_admsrc)
    barplot(range(1,27), admsrc_rankings, 'Admission Source ID', True)
    ig_rankings.append((ig_admsrc,'admission source'))

    ## hospital duration
    hospdur_e_sum = 0.0
    hospdur_rankings = []
    for i in range(14):
        if hospdur_arr[i][0] != 0:
            hospdur_rankings.append(hospdur_arr[i][1]/hospdur_arr[i][0])
            hospdur_e_sum += ((hospdur_arr[i][0]/INDEX) * entropy(hospdur_arr[i][1]/hospdur_arr[i][0]))
    ig_hospdur = readmit_e - hospdur_e_sum
    #print('The information gain found from the hospital duration feature is:', ig_hospdur)
    barplot(np.arange(1,15), hospdur_rankings, 'Hospital Duration (Days)', True)
    ig_rankings.append((ig_hospdur,'hospital duration'))

    ## number of lab procedures
    numlp_e_sum = 0.0
    numlp_rankings = []
    for i in range(14):
        if numlp_arr[i][0] != 0:
            numlp_rankings.append(numlp_arr[i][1]/numlp_arr[i][0])
            numlp_e_sum += ((numlp_arr[i][0]/INDEX)*entropy(numlp_arr[i][1]/numlp_arr[i][0]))
    ig_numlp = readmit_e - numlp_e_sum
    #print('The information gain found from the number of lab procedures feature is:', ig_numlp)
    barplot(range(14), numlp_rankings, 'Number of Lab Procedures rounded down to an interval of 10', False)
    ig_rankings.append((ig_numlp, 'number of lap procedures'))

    ## number of procedures
    numproc_e_sum = 0.0
    numproc_rankings = []
    for i in range(7):
        if numproc_arr[i][0] != 0:
            numproc_rankings.append(numproc_arr[i][1]/numproc_arr[i][0])
            numproc_e_sum += ((numproc_arr[i][0]/ INDEX)*entropy(numproc_arr[i][1]/numproc_arr[i][0]))
    ig_numproc = readmit_e - numproc_e_sum
    #print('The information gain found from the number of procedures feature is:', ig_numproc)
    barplot(range(7), numproc_rankings, 'Number of Procedures',False)
    ig_rankings.append((ig_numproc, 'number of procedures'))

    ## number of medications
    nummeds_e_sum = 0.0
    nummeds_rankings = []
    for i in range(9):
        if nummeds_arr[i][0] != 0:
            nummeds_rankings.append(nummeds_arr[i][1]/nummeds_arr[i][0])
            nummeds_e_sum += ((nummeds_arr[i][0]/INDEX) * entropy(nummeds_arr[i][1]/nummeds_arr[i][0]))
        else: nummeds_rankings.append(0)
    ig_nummeds = readmit_e - nummeds_e_sum
    #print('The information gain found from the number of medications feature is:', ig_nummeds)
    barplot(range(9), nummeds_rankings, 'Number of Medications rounded down to an interval of 10', False)
    ig_rankings.append((ig_nummeds, 'number of medications'))

    ## number of emergencies
    em_rankings = []
    for i in range(8):
        em_rankings.append(em_arr[i,1]/em_arr[i,0])
    barplot(range(8), em_rankings, 'Number of Emergencies rounded down to an interval of 10', False)


    emergencies_e_sum = 0.0
    for i in range(2):
        if emergencies_arr[i][0] != 0:
            emergencies_e_sum += ((emergencies_arr[i][0]/INDEX)*entropy(emergencies_arr[i][1]/ emergencies_arr[i][0]))
    ig_emergencies = readmit_e - emergencies_e_sum
    #print('The information gain found from the number of emergencies feature is:', ig_emergencies)
    ig_rankings.append((ig_emergencies, 'number of emergencies'))

    ## insulin levels
    insulin_e_sum = 0.0
    insulin_rankings = []
    for i in range(4):
        if insulin_arr[i][0] != 0:
            insulin_rankings.append(insulin_arr[i,1]/insulin_arr[i,0])
            insulin_e_sum += ((insulin_arr[i][0]/INDEX)*entropy(insulin_arr[i][1]/insulin_arr[i][0]))
    ig_insulin = readmit_e - insulin_e_sum
    barplot(['Up','Down','Steady','No'], insulin_rankings, 'Insulin Levels', False)
    #print('The information gain found from the insulin levels feature is:', ig_insulin)
    ig_rankings.append((ig_insulin, 'insulin levels'))
    ig_rankings.sort(key= lambda i:i[0], reverse= False)
    print(ig_rankings)

    readmitted_rankings =[]
    readmitted_rankings.append(readmitted_counter/INDEX)
    readmitted_rankings.append(1-(readmitted_counter/INDEX))
    barplot(['Readmitted', 'Not Readmitted'], readmitted_rankings, 'Y/N', False)
    # ig_results = np.array(ig_rankings)
    # fig = plt.figure(figsize=(16,12))
    # ax = fig.add_subplot(111)
    # ax.set_xticks(np.arange(len(ig_results[:,1])))
    # ax.set_xticklabels(ig_results[:,1],rotation=45, ha = 'right')
    # ax.bar(ig_results[:,1],ig_results[:,0])
    # plt.show()
    return ig_rankings

def barplot(X,Y, x_label, startsatone):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    if startsatone:
        ax.set_xticks(np.arange(1,len(X)+1))
    else:
        ax.set_xticks(np.arange(len(X)))
    ax.set_xticklabels(X, rotation=45, ha='right')
    ax.bar(X, Y)
    plt.ylabel('Perent Readmitted')
    plt.xlabel(x_label)
    plt.show()

def entropy(percentA):
    if percentA == 0 or percentA == 1:
        return 1.0
    percentB = 1-percentA
    return -(percentA*np.log2(percentA) + (percentB*np.log2(percentB)))

def createDecisionTree(dataset):
    dt = tree.DecisionTreeClassifier()
    X = dataset[1:100000, :-1]
    Y = dataset[1:100000, -1]

    Xte = np.array(dataset[100000:100001,:-1],dtype='float32')
    Yte = np.array(dataset[100000:100001,-1],dtype= 'float32')
    feature_names = ['Older than 40?', 'Escalation Admission Type?', 'Escalating Discharge Type?', 'Admission Source Newborn Related?','Time in Hospital > 7 Days', 'Lab Procedures > 60?', 'Procedures > 0?', 'Number of Medications > 40?', 'Number of Emergencies > 0?', 'Insulin levels steady?', 'Taking Diabetic Meidcations?']
    dt = dt.fit(X,Y)
    correct =0
    falseneg = 0
    falsepos = 0
    diff = 101768-100001
    for i in range(100001,101768):
        Xte = np.array(dataset[i:i+1, :-1], dtype='float32')
        Yte = np.array(dataset[i:i+1, -1], dtype='float32')
        dtp = dt.predict(Xte, Yte)
        if str(dtp) == str(Yte):
            correct += 1
        elif str(dtp) == '[0.]':
            falseneg+= 1
        else: falsepos+=1
    results = [correct/diff, falseneg/diff, falsepos/diff]
    barplot(['Correct','False Negative', 'False Positive'],results,'Percent of Predictions', False)
    print(correct/diff)
    tree.plot_tree(dt)
    data = tree.export_graphviz(dt,out_file=None, feature_names = feature_names, filled=True, rounded= True, class_names= ['Not Readmitted','Readmitted'])
    graph = graphviz.Source(data)
    graph.render("dtree")


    #dtp = dt.predict(Xte,Yte)
if __name__ == "__main__":
    main()