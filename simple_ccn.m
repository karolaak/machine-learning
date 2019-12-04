digitDatasetPath = fullfile ('C:\Users\Damian\Desktop\animals\');
images = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


[trainingImages,validationImages] = splitEachLabel(images,0.7,'randomize');
numTrainImages = numel(trainingImages.Labels);

labelCount = countEachLabel(images)

img = readimage(images,1);
size(img)

idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end

%numTrainFiles = 300;
%[imdsTrain,imdsValidation] = splitEachLabel(images,numTrainFiles,'randomize');

layers = [
 imageInputLayer([27 27 3])
    
    convolution2dLayer(3,112,'Stride',4)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,227,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.02, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationImages, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','parallel');

netTransfer = trainNetwork(trainingImages,layers,options);
predictedLabels = classify(netTransfer,validationImages);

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(validationImages,idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end

valLabels = validationImages.Labels;
%accuracy = sum(YPred == YValidation)/numel(YValidation)
accuracy = sum(predictedLabels == valLabels)/numel(valLabels);
