%%
clc
clear
close
rng(1)
ori_input = importdata('data.xlsx');%输入数据
ori_real= importdata('label.xlsx');%真实标签]
input=ori_input';%输入数据转置，因为其将一个列向量作为一个样本
real=ori_real';
[norm_input,ps]=mapminmax(input);%输入归一化
[norm_real,ts]=mapminmax(real);%输出归一化
TF1='logsig';TF2='logsig';TF3='tansig';%神经网络结构
net=newff(norm_input,norm_real,[10,255,1],{TF1 TF2 TF3});%创建bp神经网络，隐含层为10*255*1 
net.trainFcn='traingd';%注意要写在前面，卸载后面的话，接下来的设置都没用了     !!!这里注意，如果将traingd改为trainlm则仅需要训练10步以内就你可以，非常快
net.trainParam.epochs=5000;%训练次数
net.trainParam.goal=3e-20;%误差精度
net.trainParam.lr=0.01;%学习率
net.trainParam.mc=0.9;%动量因子为0.9
net.trainParam.min_grad=1e-19;%停止更新时梯度
net.layers{1}.initFcn = 'initwb';%以下为初始化参数，输入层是用自带方式初始化
net.inputWeights{1,1}.initFcn = 'rands';%第一层weight使用随机初始化， 下同
net.biases{1,1}.initFcn = 'rands';
net.biases{2,1}.initFcn = 'rands';
net = init(net);
[net,tr]=train(net,norm_input,norm_real);%开始训练
%绘制误差曲线
figure
plot(tr.perf)
xlabel('episode');
ylabel('MSE');
%保存参数
save('model','net')
%%
%当测试时，打开只运行这一节就可以
load('-mat','model')

% %当测试其他文件时时，将下面这两行代码注释打开，并将新的测试数据并命为test.xlsx
% test_input = importdata("test.xlsx");%输入数据
% test_data = test_input';

%得到测试训练用的数据
test_data = input;

%开始预测
[data,~]=mapminmax(test_data);
anewn=sim(net,data);
[anew,~]=mapminmax('reverse',anewn,ts);%输出所有样本的预测输出

%预测值记录到excell
filename = 'prediction.xlsx';
A = anew;
sheet = 1;
xlswrite(filename,A,sheet)

%画图
figure
plot(anew)
hold on 
plot(real)
xlabel('数据对比');