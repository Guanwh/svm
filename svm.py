import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import random
import pickle as pkl
import datetime
def read_data(path):
    i = 1
    coord_x=[]
    coord_y=[]

    with open(path) as f:
        for line in f.readlines():

            if str(line[:4])=="samp":
                i = i+15
                continue
            list_str = line.split('=')
            coord = list_str[1].strip('\n').split(',')
            coord_x.append(int(coord[0][1:]))
            coord_y.append(int(coord[1][:-1]))
#    print('---------------',zip(coord_x,coord_y))
    sample_tuple = list(zip(coord_x,coord_y))
    #sample_list = [list(each) for each in sample_tuple]

    list_jump = []
    for index,key in enumerate(sample_tuple):
        if index ==0:
            continue
        if (index+1)%14==0:
            list_jump.append(sample_tuple[index-13:index+1])
    print('len==========',len(sample_tuple))
    X = []
    for each in list_jump:
        pic = []
        for i in each:
            pic.append(i[0])
            pic.append(i[1])
        X.append(pic)
    #return sample_tuple
    return X
def svm_train(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.03, random_state=33)
    tuned_parameters=[{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]}]
    #tuned_parameters=[{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]},
    #                  {'kernel':['linear'],'C':[1,10,100,1000]}]
    scores=['precision','recall']
    for score in scores:
        print('#Tuning hyper-parameters for %s' % score)
        #调用GridSearchC，将SVC（），tuned_parameters,cv=5,还有scoring传递进去
        model=GridSearchCV(svm.SVC(),tuned_parameters,cv=5,scoring='%s_macro' % score)
        #用训练集训练这个学习器
        print('training...')
        model.fit(x_train, y_train)
        print("Best parameters set found on development set:")
        print()
        #再调用clf.best_params_就能直接得到最好的参数搭配结果
        print(model.best_params_)

        print()
        print("Grid scores on developmebnt set:")
        print()
        means=model.cv_results_['mean_test_score']
        stds=model.cv_results_['std_test_score']
        #看一下具体的参数建不同数值的组合后的到的分数是多少
        for mean,std,params in zip(means,stds,model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean,std * 2,params))
        print()

        print("detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evalution set.")
        print()
        time_start=datetime.datetime.now()
        y_true,y_pred =y_test, model.predict(x_test)
        time_end=datetime.datetime.now()
        print('all_second_time:',(time_end-time_start).total_seconds())
        #打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true,y_pred))

        print()
    #score = model.score(x_test,y_test)

    with open('D:\Pythonwork35\svm\data\model.pkl', 'wb') as f1:
        pkl.dump(model, f1)
    with open('D:\Pythonwork35\svm\data\pre.pkl', 'wb') as f2:
        pkl.dump(y_pred, f2)
    with open('D:\Pythonwork35\svm\data\y_test.pkl', 'wb') as f3:
        pkl.dump(y_test, f3)

    sumnum = 0
    for index, key in enumerate(y_pred):
        if key == y_test[index]:
            sumnum = sumnum + 1
    print("acc=", sumnum / len(y_pred))

if __name__ =="__main__":
    list_jump = read_data('D:\Pythonwork35\svm\data\jump.txt')
#    list_lie = read_data('../data/lie.txt')
    list_sit = read_data('D:\Pythonwork35\svm\data\jump.txt')
    jump_y = [0]*len(list_jump)
#    lie_y = [1]*len(list_lie)
    sit_y = [2]*len(list_sit)
#    x = list_jump+list_lie
#    y = jump_y+lie_y
    x = list_jump + list_sit
    y = jump_y + sit_y
#    print('xxxxxxxxxxxx===',x)
#    print('yyyyyyyyyyyy===',y)
    tuple_x_y = list(zip(x,y))
    random.shuffle(tuple_x_y)
    x_axis = []
    y_label = []
    for each in tuple_x_y:
        x_axis.append(each[0])
        y_label.append(each[1])
    print('x_axis======',x_axis)
    print(len(x_axis))
    print(len(y_label))
    svm_train(x_axis,y_label)


