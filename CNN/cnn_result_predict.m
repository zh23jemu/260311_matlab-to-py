function [feature_real,output_real]=cnn_result_predict(Netmod,inputdata,layer)
trainTEST_original=inputdata;%原始数据
for i=1:1:size(trainTEST_original,1)
trainTEST_original_4D(:,:,:,i)=trainTEST_original(i,:);
end
TEST_4D = classify(Netmod,trainTEST_original_4D);

output_real=TEST_4D;

output_real=double(TEST_4D);

feature = activations(Netmod,trainTEST_original_4D,layer);
for i=1:1:size(feature,4)
for j=1:1:size(feature,3)
 feature2(:,j,i)=feature(:,1,j,i); 
end
 end
for i=1:1:size(feature,4)
featurep=feature2(:,:,i);
 feature_real(i,:)=featurep(:);
end

end

 