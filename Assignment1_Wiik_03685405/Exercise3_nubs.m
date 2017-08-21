function [  ] = Exercise3_nubs( gesture, k )
    v = [0.08; 0.05; 0.02];
    %Looping through all traces and all points to assign x, y and z-values
    %to 'coordinates'. 'counter_points' is used as index for all points in
    %all clusters.
    coordinates = zeros(size(gesture,1)*size(gesture,2),size(gesture,3));
    counter_points = 0;
    for trace = 1:size(gesture,2) %1:10
        for point = 1:size(gesture,1) %1:60
            counter_points = counter_points + 1;
            coordinates(counter_points,:) = gesture(point,trace,:);
        end
    end
    %Creating a 'classes' array which holds clusters in first column and
    %the corresponding total distortions 'tot_dists' in second column.
    classes = {coordinates};
    %Creating and concatenating centroids into the vector 'y'. 
    y = [];
    y = [y mean(classes{1,1})];
    %Looping through each split.
    for split = 1:k-1
        tot_dists = zeros(size(classes,1),1);
        for class = 1:size(classes,1) 
            %Distance matrix 'distances' is reset to size of current class
            %and the euclidian distance between all points and centroids
            %are assigned to it.
            distances = zeros(size(classes{class,1},1),1);
            for point = 1:size(classes{class,1},1) 
               distances(point,:) = sqrt(((classes{class,1}(point,1))-(y(class,1)))^2 + ...
                                        ((classes{class,1}(point,2))-(y(class,2)))^2 + ...
                                        ((classes{class,1}(point,3))-(y(class,3)))^2);
            end
            classes{class,2} = sum(distances);
            tot_dists(class) = sum(distances);
        end
        %Splitting class with largest distortion into two subclasses
        %'subclass_a' and 'subclass_b'. 
        largest_distortion_index = find(tot_dists==max(tot_dists,[],1));
        y_a = y(largest_distortion_index,:)+v';
        y_b = y(largest_distortion_index,:)-v';
        largest_distortion_class = zeros(size(classes{largest_distortion_index,1},1),2);
        for point = 1:size(classes{largest_distortion_index,1},1)
            largest_distortion_class(point,1) = sqrt(((classes{largest_distortion_index,1}(point,1))-(y_a(1)))^2 + ...
                                           ((classes{largest_distortion_index,1}(point,2))-(y_a(2)))^2 +  ...
                                           ((classes{largest_distortion_index,1}(point,3))-(y_a(3)))^2);
            largest_distortion_class(point,2) = sqrt(((classes{largest_distortion_index,1}(point,1))-(y_b(1)))^2 + ...
                                           ((classes{largest_distortion_index,1}(point,2))-(y_b(2)))^2 + ...
                                           ((classes{largest_distortion_index,1}(point,3))-(y_b(3)))^2);
        end
        subclass_a_indices = find(largest_distortion_class(:,1) <= largest_distortion_class(:,2));
        subclass_b_indices = find(largest_distortion_class(:,1) > largest_distortion_class(:,2));

        subclass_a = classes{largest_distortion_index,1}(subclass_a_indices,:);
        subclass_b = classes{largest_distortion_index,1}(subclass_b_indices,:);
        %Class with largest distortion is overwritten by subclass_a.
        %Subclass_b is added as new class.
        classes{largest_distortion_index,1} = subclass_a;
        classes{split+1,1} = subclass_b;
        %Update Centroids
        y(largest_distortion_index,:) = mean(subclass_a,1);
        y(split+1,:) = mean(subclass_b,1);
    end
    
    figure(1);
    hold on;
    scatter3(classes{1,1}(:,1),classes{1,1}(:,2),classes{1,1}(:,3),'blue')
    scatter3(classes{2,1}(:,1),classes{2,1}(:,2),classes{2,1}(:,3),'black')
    scatter3(classes{3,1}(:,1),classes{3,1}(:,2),classes{3,1}(:,3),'red')
    scatter3(classes{4,1}(:,1),classes{4,1}(:,2),classes{4,1}(:,3),'green')
    scatter3(classes{5,1}(:,1),classes{5,1}(:,2),classes{5,1}(:,3),'magenta')
    scatter3(classes{6,1}(:,1),classes{6,1}(:,2),classes{6,1}(:,3),'yellow')
    scatter3(classes{7,1}(:,1),classes{7,1}(:,2),classes{7,1}(:,3),'cyan')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    title('Nubs')
end

