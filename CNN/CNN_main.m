clear all
data_standar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  训练网络  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
traindata=[d01bianhuan1;d02bianhuan1;d04bianhuan1;d05bianhuan1;d06bianhuan1;d07bianhuan1;d08bianhuan1;...
           d10bianhuan1;d11bianhuan1;d12bianhuan1;d13bianhuan1;d14bianhuan1;d16bianhuan1;d17bianhuan1;d18bianhuan1;d19bianhuan1;...
           d20bianhuan1];%原始训练样本
Labels=1*[repmat([1],length(d01bianhuan1),1);repmat([2],length(d02bianhuan1),1);...
          repmat([3],length(d04bianhuan1),1);repmat([4],length(d05bianhuan1),1);repmat([5],length(d06bianhuan1),1);...
          repmat([6],length(d07bianhuan1),1);repmat([7],length(d08bianhuan1),1);repmat([8],length(d10bianhuan1),1);...
          repmat([9],length(d11bianhuan1),1);repmat([10],length(d12bianhuan1),1);repmat([11],length(d13bianhuan1),1);...
          repmat([12],length(d14bianhuan1),1);repmat([13],length(d16bianhuan1),1);repmat([14],length(d17bianhuan1),1);...
          repmat([15],length(d18bianhuan1),1);repmat([16],length(d19bianhuan1),1);repmat([17],length(d20bianhuan1),1);];%原始训练标签
outputCategories = 17;%故障类数
[CNNNet1,train_accuracy,net_output_real]=cnnlayer(traindata,Labels,outputCategories,0.03,10);%0.03
%%%%%%%%%%%%%%%%%%%%%%%%%%%%   测试网络   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_data1=d01_tebianhuan1(1:2000,:);test_data2=d02_tebianhuan1(1:2000,:);test_data4=d04_tebianhuan1(1:2000,:);
test_data5=d05_tebianhuan1(1:2000,:);test_data6=d06_tebianhuan1(1:247,:);test_data7=d07_tebianhuan1(1:2000,:);test_data8=d08_tebianhuan1(1:2000,:);
test_data10=d10_tebianhuan1(1:2000,:);test_data11=d11_tebianhuan1(1:2000,:);test_data12=d12_tebianhuan1(1:2000,:);
test_data13=d13_tebianhuan1(1:2000,:);test_data14=d14_tebianhuan1(1:2000,:);test_data16=d16_tebianhuan1(1:2000,:);
test_data17=d17_tebianhuan1(1:2000,:);test_data18=d18_tebianhuan1(1:2000,:);test_data19=d19_tebianhuan1(1:2000,:);
test_data20=d20_tebianhuan1(1:2000,:);
%test_data21=d21_tebianhuan1;

test_label1=categorical([1*ones(2000);]);test_label2=categorical([2*ones(2000);]);
test_label4=categorical([3*ones(2000);]);test_label5=categorical([4*ones(2000);]);test_label6=categorical([5*ones(247,1);]);
test_label7=categorical([6*ones(2000);]);test_label8=categorical([7*ones(2000);]);test_label10=categorical([8*ones(2000);]);
test_label11=categorical([9*ones(2000);]);test_label12=categorical([10*ones(2000);]);test_label13=categorical([11*ones(2000);]);
test_label14=categorical([12*ones(2000);]);test_label16=categorical([13*ones(2000);]);test_label17=categorical([14*ones(2000);]);
test_label18=categorical([15*ones(2000);]);test_label19=categorical([16*ones(2000);]);test_label20=categorical([17*ones(2000);]);

tic
[feature_layer_real1,output_real1]=cnn_result_predict(CNNNet1,test_data1,6);
[feature_layer_real2,output_real2]=cnn_result_predict(CNNNet1,test_data2,6);
[feature_layer_real4,output_real4]=cnn_result_predict(CNNNet1,test_data4,6);
[feature_layer_real5,output_real5]=cnn_result_predict(CNNNet1,test_data5,6);
[feature_layer_real6,output_real6]=cnn_result_predict(CNNNet1,test_data6,6);
[feature_layer_real7,output_real7]=cnn_result_predict(CNNNet1,test_data7,6);
[feature_layer_real8,output_real8]=cnn_result_predict(CNNNet1,test_data8,6);
[feature_layer_real10,output_real10]=cnn_result_predict(CNNNet1,test_data10,6);
[feature_layer_real11,output_real11]=cnn_result_predict(CNNNet1,test_data11,6);
[feature_layer_real12,output_real12]=cnn_result_predict(CNNNet1,test_data12,6);
[feature_layer_real13,output_real13]=cnn_result_predict(CNNNet1,test_data13,6);
[feature_layer_real14,output_real14]=cnn_result_predict(CNNNet1,test_data14,6);
[feature_layer_real16,output_real16]=cnn_result_predict(CNNNet1,test_data16,6);
[feature_layer_real17,output_real17]=cnn_result_predict(CNNNet1,test_data17,6);
[feature_layer_real18,output_real18]=cnn_result_predict(CNNNet1,test_data18,6);
[feature_layer_real19,output_real19]=cnn_result_predict(CNNNet1,test_data19,6);
[feature_layer_real20,output_real20]=cnn_result_predict(CNNNet1,test_data20,6);
toc

accuracy_t1 = sum(output_real1(1:2000) == 1)/2000
accuracy_t2 = sum(output_real2(1:2000) == 2)/2000

accuracy_t4 = sum(output_real4(1:2000) == 3)/2000
accuracy_t5 = sum(output_real5(1:2000) == 4)/2000
accuracy_t6 = sum(output_real6(1:24) == 5)/247
accuracy_t7 = sum(output_real7(1:2000) == 6)/2000
accuracy_t8 = sum(output_real8(1:2000) == 7)/2000

accuracy_t10 = sum(output_real10(1:2000) == 8)/2000
accuracy_t11 = sum(output_real11(1:2000) == 9)/2000
accuracy_t12 = sum(output_real12(1:2000) == 10)/2000
accuracy_t13 = sum(output_real13(1:2000) == 11)/2000
accuracy_t14 = sum(output_real14(1:2000) == 12)/2000

accuracy_t16 = sum(output_real16(1:2000) == 13)/2000
accuracy_t17 = sum(output_real17(1:2000) == 14)/2000
accuracy_t18 = sum(output_real18(1:2000) == 15)/2000
accuracy_t19 = sum(output_real19(1:2000) == 16)/2000
accuracy_t20 = sum(output_real20(1:2000) == 17)/2000

accuracyp1=[1;2;4;5;6;7;8;10;11;12;13;14;16;17;18;19;20];
accuracyp2=[accuracy_t1;accuracy_t2;accuracy_t4;accuracy_t5;accuracy_t6;...
    accuracy_t7;accuracy_t8;accuracy_t10;accuracy_t11;accuracy_t12;...
    accuracy_t13;accuracy_t14;accuracy_t16;accuracy_t17;accuracy_t18;...
    accuracy_t19;accuracy_t20;];
accuracy=[accuracyp1,accuracyp2]
mean(accuracyp2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  热值图   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:17

Heat(1,i)= (sum(output_real1(1:2000) == i*ones(2000,1)))/2000;

Heat(2,i)= (sum(output_real2(1:2000) == i*ones(2000,1)))/2000;

Heat(3,i)= (sum(output_real4(1:2000) == i*ones(2000,1)))/2000;
Heat(4,i)= (sum(output_real5(1:2000) == i*ones(2000,1)))/2000;
Heat(5,i)= (sum(output_real6(1:247) == i*ones(247,1)))/247;
Heat(6,i)= (sum(output_real7(1:2000) == i*ones(2000,1)))/2000;
Heat(7,i)= (sum(output_real8(1:2000) == i*ones(2000,1)))/2000;

Heat(8,i)= (sum(output_real10(1:2000) == i*ones(2000,1)))/2000;
Heat(9,i)= (sum(output_real11(1:2000) == i*ones(2000,1)))/2000;
Heat(10,i)= (sum(output_real12(1:2000) == i*ones(2000,1)))/2000;
Heat(11,i)= (sum(output_real13(1:2000) == i*ones(2000,1)))/2000;
Heat(12,i)= (sum(output_real14(1:2000) == i*ones(2000,1)))/2000;

Heat(13,i)= (sum(output_real16(1:2000) == i*ones(2000,1)))/2000;
Heat(14,i)= (sum(output_real17(1:2000) == i*ones(2000,1)))/2000;
Heat(15,i)= (sum(output_real18(1:2000) == i*ones(2000,1)))/2000;
Heat(16,i)= (sum(output_real19(1:2000) == i*ones(2000,1)))/2000;
Heat(17,i)= (sum(output_real20(1:2000) == i*ones(2000,1)))/2000;

end


figure(1)
h.ColorbarVisible = 'off';
xvalues = {'F1','F2','F4','F5','F6','F7','F8','F10','F11','F12','F13'...
    ,'F14','F16','F17','F18','F19','F20'};
yvalues = {'F1','F2','F4','F5','F6','F7','F8','F10','F11','F12','F13'...
    ,'F14','F16','F17','F18','F19','F20'};
hh=round(Heat,2);
H=heatmap(xvalues,yvalues,hh);
h.ColorbarVisible = 'off';
%text(Heat,cellstr(str));





