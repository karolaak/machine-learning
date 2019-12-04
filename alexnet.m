digitDatasetPath = fullfile('C:\Users\Damian\Desktop\karola\proj\animals\');
images = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%img = readimage(trainDigitData,1);
%digitDatasetPath = fullfile('C:\Users\ckacz\OneDrive\Pulpit\Karolcia\CV\c_dogs\');
%testDigitData = imageDatastore(digitDatasetPath, ...
 %   'IncludeSubfolders',true,'LabelSource','foldernames');

%[trainingImages, validationImages] = splitEachLabel(images,0.7,'randomized');
%[trainingImages,validationImages] = splitEachLabel(images,numTrainFiles,'randomize');
[trainingImages,validationImages] = splitEachLabel(images,0.9,'randomize');
numTrainImages = numel(trainingImages.Labels);


idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end

net = alexnet;
net.Layers
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImages.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',2,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);

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
accuracy = mean(predictedLabels == valLabels)/numel(valLabels);
