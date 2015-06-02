function [ model, Value ] = GenerateRUSBoostModel( X, Y )
%GENERATERUSBOOSTMODEL Generate RUSBoost Model and display its performance
%based on the parameters defined below. 
    part = cvpartition(Y,'holdout',0.3);
    istrain = training(part);
    istest = test(part);
    

    t = templateTree();
    rusTree = fitensemble(X(istrain,:),Y(istrain),'RUSBoost',300,t,...
        'LearnRate',0.1);

%     figure;
%     plot(loss(rusTree,X(istest,:),Y(istest),'mode','cumulative'));
%     grid on;
%     xlabel('Number of trees');
%     ylabel('Test classification error');

    Yfit = predict(rusTree,X(istest,:));
    tab = tabulate(Y(istest));
  %  bsxfun(@rdivide,confusionmat(Y(istest),Yfit),tab(:,2))*100;

    model = rusTree;
    Value = bsxfun(@rdivide,confusionmat(Y(istest),Yfit),tab(:,2))*100;
  %t = templateTree('MinLeafSize',10);
  %rusTree = fitensemble(X,Y, 'RUSBoost', 300, t,  'LearnRate', 0.1, 'kFold', 40);
  %model = rusTree; 
  %Value = kfoldLoss(rusTree, 'lossfun', @lossfun);
  
end

