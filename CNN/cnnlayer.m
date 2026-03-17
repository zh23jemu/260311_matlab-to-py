function [CNNNet,accuracy,output_real]=cnnlayer(inputdata,labeldata,numImageCategories,learnrate,Epoc)
%%%%%%%%%%%%%%%%%    训练数据   随机化  格式转换  %%%%%%%%%%%%%%%%%%%%%%%%
traindata_original=inputdata;%原始数据
for i=1:1:size(traindata_original,1)
traindata_original_4D(:,:,:,i)=traindata_original(i,:);%修改训练样本格式
end
labeldata_sta=categorical(labeldata);%修改训练标签格式
order = randperm(size(traindata_original_4D,4));%随机化参数
traindata=traindata_original_4D(:,:,:,order(1:end));%训练数据随机化
traindataLabels=labeldata_sta(order(1:end),:);%训练标签随机化
%%%%%%%%%%%         网络参数    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imageSize = [size(traindata_original,2) 1 1];
%%% Convolutional layer parameters
filterSize1 = [2 1]; numFilters1 = 64;
filterSize2 = [3 1]; numFilters2 = 32;
filterSize3 = [4 1]; numFilters3 = 16;

fullyConnectedLayernum1=400;
%fullyConnectedLayernum2=250;
%fullyConnectedLayernum3=150;
%fullyConnectedLayernum4=75;
%fullyConnectedLayernum5=37;
%fullyConnectedLayernum6=19;
%fullyConnectedLayernum7=15;
%fullyConnectedLayernum8=10;
fullyConnectedLayernum2=numImageCategories;

% Convolutional layer parameters

%inputLayer = imageInputLayer(imageSize,"Name","input1");%
layers = [
    
imageInputLayer(imageSize,"Name","input1");
convolution2dLayer(filterSize1, numFilters1,"Name","conv1", 'Padding', 1)
tanhLayer("Name","tan1")
%reluLayer("Name","relu1")

convolution2dLayer(filterSize2, numFilters2, "Name","conv2",'Padding', 1)
tanhLayer("Name","tan2")
%reluLayer("Name","relu2")

%%maxPooling2dLayer(3, 'Stride',1)
convolution2dLayer(filterSize3,  numFilters3, "Name","conv3",'Padding', 1)
tanhLayer("Name","tan3")
%reluLayer("Name","relu3")

fullyConnectedLayer(fullyConnectedLayernum1,"Name","fc1")
tanhLayer("Name","tan4")

fullyConnectedLayer(fullyConnectedLayernum2,"Name","fc9")

softmaxLayer("Name","sm1")

classificationLayer("Name","cf1")
];


% Set the network training options
opts = trainingOptions('sgdm', ...
   'InitialLearnRate', learnrate, ...%0.01
  'MaxEpochs', Epoc, ...%400 % 
  'MiniBatchSize', 300, ...%300
'Verbose', true,...
 'VerboseFrequency',1, ...
'Plots','training-progress',...
 'ExecutionEnvironment', 'cpu');

plot(layerGraph(layers));

CNNNet = trainNetwork(traindata, traindataLabels, layers, opts);
% rescale the weights to the range [0, 1] for better visualization
%w = rescale(w);
%figure
%montage(w)
%for i=1:1:size(w,4)
 %   ww(:,i)=w(:,:,:,i)
%end
% Run the network on the test set.

output_real = classify(CNNNet, traindata);
accuracy = sum(output_real == traindataLabels)/numel(traindataLabels);

