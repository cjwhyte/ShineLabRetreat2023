
close all
clear


%% Perceptual decision making

activity = load('activity_PD1.csv');
activity = reshape(activity,[50,50,1000]);

conditon = load('trial_condition_PD1.csv');

% sort

cond_idx = conditon == 0;

activity_cond1 = activity(:,:,conditon == 0);
activity_cond2 = activity(:,:,conditon == 1);


mean_cond1 = mean(activity_cond1,3);
mean_cond2 = mean(activity_cond2,3);

% PCA

concat_activity = permute(activity,[1,3,2]);
concat_activity = reshape(concat_activity, [50*1000,50]);
concat_activity = zscore(concat_activity);


[vecs,vals,~,~,explained] = pca(concat_activity);

for pc = 1:3
    for t = 1:size(activity,1)
        projection_c1(t,pc) = mean_cond1(t,:)*vecs(:,pc);
        projection_c2(t,pc) = mean_cond2(t,:)*vecs(:,pc);
    end 
end 

figure(1)
hold on
plot(mean_cond1,'k','linewidth',3)

figure(3)
hold on
plot3(projection_c1(:,1),projection_c1(:,2),projection_c1(:,3),'b','linewidth',4)
plot3(projection_c2(:,1),projection_c2(:,2),projection_c2(:,3),'r','linewidth',4)

return
%% WM


activity = load('activity_WM1.csv');
activity = reshape(activity,[50,50,1000]);

conditon = load('trial_condition_WM1.csv');

activity_cond1 = activity(:,:,conditon == 0);
activity_cond2 = activity(:,:,conditon == 1);
activity_cond3 = activity(:,:,conditon == 2);
activity_cond4 = activity(:,:,conditon == 3);

mean_cond1 = mean(activity_cond1,3);
mean_cond2 = mean(activity_cond2,3);
mean_cond3 = mean(activity_cond3,3);
mean_cond4 = mean(activity_cond4,3);

% PCA

concat_activity = permute(activity,[1,3,2]);
concat_activity = reshape(concat_activity, [50*1000,50]);
concat_activity = zscore(concat_activity);

[vecs,vals,~,~,explained] = pca(concat_activity);

for pc = 1:3
    for t = 1:size(activity,1)
        projection_c1(t,pc) = mean_cond1(t,:)*vecs(:,pc);
        projection_c2(t,pc) = mean_cond2(t,:)*vecs(:,pc);
        projection_c3(t,pc) = mean_cond3(t,:)*vecs(:,pc);
        projection_c4(t,pc) = mean_cond4(t,:)*vecs(:,pc);
    end 
end 

figure(1)
hold on
plot(mean_cond1,'k','linewidth',3)
plot(mean_cond2,'b','linewidth',3)

figure(2)
hold on
plot(mean_cond3,'k','linewidth',3)
plot(mean_cond4,'b','linewidth',3)

figure(3)
hold on
plot3(projection_c1(:,1),projection_c1(:,2),projection_c1(:,3),'b','linewidth',4)
plot3(projection_c2(:,1),projection_c2(:,2),projection_c2(:,3),'b--','linewidth',4)
plot3(projection_c3(:,1),projection_c3(:,2),projection_c3(:,3),'r','linewidth',4)
plot3(projection_c4(:,1),projection_c4(:,2),projection_c4(:,3),'r--','linewidth',4)


