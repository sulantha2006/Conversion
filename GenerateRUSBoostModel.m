function [ model, Value ] = GenerateRUSBoostModel( X, Y )
%GENERATERUSBOOSTMODEL Generate RUSBoost Model and display its performance
%based on the parameters defined below. 
%    part = cvpartition(Y,'holdout',0.3);
%    istrain = training(part);
%    istest = test(part);
    

%    t = templateTree();
%    rusTree = fitensemble(X(istrain,:),Y(istrain),'RUSBoost',300,t,...
%        'LearnRate',0.1);

%     figure;
%     plot(loss(rusTree,X(istest,:),Y(istest),'mode','cumulative'));
%     grid on;
%     xlabel('Number of trees');
%     ylabel('Test classification error');

%    Yfit = predict(rusTree,X(istest,:));
%    tab = tabulate(Y(istest));
  %  bsxfun(@rdivide,confusionmat(Y(istest),Yfit),tab(:,2))*100;

%   model = rusTree;
%    Value = bsxfun(@rdivide,confusionmat(Y(istest),Yfit),tab(:,2))*100;
  %t = templateTree('MinLeafSize',10);
  %rusTree = fitensemble(X,Y, 'RUSBoost', 300, t,  'LearnRate', 0.1, 'kFold', 40);
  %model = rusTree; 
  %Value = kfoldLoss(rusTree, 'lossfun', @lossfun);
  t1 = templateTree('MinLeafSize',10);
  rusTree1 = fitensemble(X, Y, 'RUSBoost', 250, t1, 'LearnRate', 0.3, 'kFold', 5);
  [Set1Values, Score1] = kfoldPredict(rusTree1);
  PosteriorProb1 = double(bsxfun(@eq, Score1, max(Score1, [], 2)));
  YFit1 = PosteriorProb1(:,2);
  tab = tabulate(Y);
  Value = bsxfun(@rdivide,confusionmat(Y,YFit1),tab(:,2))*100;
  Value = confusionmat(Y, YFit1);
  model = rusTree1;
end

