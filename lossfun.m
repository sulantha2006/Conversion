function loss = lossfun(C,S,W,COST)
    S2 = double(bsxfun(@eq, S, max(S, [], 2)));
    Class0Indexes = find(C(:,1));
    Class0Error = mean(S2(Class0Indexes, 2));
    
    Class1Indexes = find(C(:,2));
    Class1Error = 3*mean(S2(Class1Indexes, 1));
    
    
    
    loss = (Class0Error + Class1Error)/4;
end
