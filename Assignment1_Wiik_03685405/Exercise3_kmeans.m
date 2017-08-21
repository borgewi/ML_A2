function [] = Exercise3_kmeans( gesture, init_cluster, k ) 
    %Renaming init_cluster so that it makes more sense in later iterations.
    centroids = init_cluster;
    %Pre-allocating matrices for speed improvements.
    dist = zeros(size(centroids,1),size(gesture,1),size(gesture,2));
    cluster_indices = zeros(size(gesture,1),size(gesture,2));
    cluster_values = zeros(size(gesture,1),size(gesture,2));
    %decrement is initially set to 1 so the while loop may be entered.
    %previous_distortion is set to 0 for initialization purposes.
    decrement = 1;
    previous_distortion = 0;
    %Looping until error is small enough
    while abs(decrement) > 10e-6
        % Find distance matrix 'dist'.
        for trace = 1:size(gesture,2) 
            for point = 1:size(gesture,1) 
                for cluster = 1:size(centroids,1) 
                    dist(cluster,point,trace) = sqrt(((gesture(point,trace,1))-(centroids(cluster,1)))^2 + ...
                          ((gesture(point,trace,2))-(centroids(cluster,2)))^2 + ...
                          ((gesture(point,trace,3))-(centroids(cluster,3)))^2);
                end
            end 
        end
        % Assign coordinates to clusters
        for trace_index = 1:size(gesture,2)
            for point_index = 1:size(gesture,1)
                [V,I] = min(dist(:,point_index,trace_index));
                cluster_indices(point_index,trace_index) = I;
                cluster_values(point_index,trace_index) = V;
            end
        end
        % Assign coordinates into color clusters. Using squeeze function to
        % remove unnecessary third dimension in cluster matrices.
        cluster_blue = []; 
        cluster_black = []; 
        cluster_red = []; 
        cluster_green = []; 
        cluster_magenta = []; 
        cluster_yellow = []; 
        cluster_cyan= [];

        cluster_blue_dist = [];
        cluster_black_dist = [];
        cluster_red_dist = [];
        cluster_green_dist = [];
        cluster_magenta_dist = [];
        cluster_yellow_dist = [];
        cluster_cyan_dist = [];

        [blue_row, blue_col] = find(cluster_indices == 1);
        for point = 1:size(blue_col,1)
            cluster_blue = [cluster_blue; gesture(blue_row(point,1),blue_col(point,1),:)];
            cluster_blue_dist = [cluster_blue_dist; cluster_values(blue_row(point,1),blue_col(point,1))];
        end
        cluster_blue = squeeze(cluster_blue);
        [black_row, black_col] = find(cluster_indices == 2);
        for point = 1:size(black_col,1)
            cluster_black = [cluster_black; gesture(black_row(point,1),black_col(point,1),:)];
            cluster_black_dist = [cluster_black_dist; cluster_values(black_row(point,1),black_col(point,1))];
        end
        cluster_black = squeeze(cluster_black);
        [red_row, red_col] = find(cluster_indices == 3);
        for point = 1:size(red_col,1)
            cluster_red = [cluster_red; gesture(red_row(point,1),red_col(point,1),:)];
            cluster_red_dist = [cluster_red_dist; cluster_values(red_row(point,1),red_col(point,1))];
        end
        cluster_red = squeeze(cluster_red);
        [green_row, green_col] = find(cluster_indices == 4);
        for point = 1:size(green_col,1)
            cluster_green = [cluster_green; gesture(green_row(point,1),green_col(point,1),:)];
            cluster_green_dist = [cluster_green_dist; cluster_values(green_row(point,1),green_col(point,1))];
        end
        cluster_green = squeeze(cluster_green);
        [magenta_row, magenta_col] = find(cluster_indices == 5);
        for point = 1:size(magenta_col,1)
            cluster_magenta = [cluster_magenta; gesture(magenta_row(point,1),magenta_col(point,1),:)];
            cluster_magenta_dist = [cluster_magenta_dist; cluster_values(magenta_row(point,1),magenta_col(point,1))];
        end
        cluster_magenta = squeeze(cluster_magenta);
        [yellow_row, yellow_col] = find(cluster_indices == 6);
        for point = 1:size(yellow_col,1)
            cluster_yellow = [cluster_yellow; gesture(yellow_row(point,1),yellow_col(point,1),:)];
            cluster_yellow_dist = [cluster_yellow_dist; cluster_values(yellow_row(point,1),yellow_col(point,1))];
        end
        cluster_yellow = squeeze(cluster_yellow);
        [cyan_row, cyan_col] = find(cluster_indices == 7);
        for point = 1:size(cyan_col,1)
            cluster_cyan = [cluster_cyan; gesture(cyan_row(point,1),cyan_col(point,1),:)];
            cluster_cyan_dist = [cluster_cyan_dist; cluster_values(cyan_row(point,1),cyan_col(point,1))];   
        end
        cluster_cyan = squeeze(cluster_cyan);
        % Calculate distortion and decrement
        current_distortion = sum(cluster_blue_dist)+sum(cluster_black_dist)+sum(cluster_red_dist)+sum(cluster_green_dist)+ ...
                             sum(cluster_magenta_dist)+sum(cluster_yellow_dist)+sum(cluster_cyan_dist);
        decrement = abs(previous_distortion - current_distortion);
        previous_distortion = current_distortion;
        % Update cluster centroids
        centroids = [mean(cluster_blue); mean(cluster_black); mean(cluster_red); mean(cluster_green); ...
                       mean(cluster_magenta); mean(cluster_yellow); mean(cluster_cyan)];
    end

    figure(1);
    scatter3(cluster_blue(:,1),cluster_blue(:,2),cluster_blue(:,3),'blue')
    hold;
    scatter3(cluster_black(:,1),cluster_black(:,2),cluster_black(:,3),'black')
    scatter3(cluster_red(:,1),cluster_red(:,2),cluster_red(:,3),'red')
    scatter3(cluster_green(:,1),cluster_green(:,2),cluster_green(:,3),'green')
    scatter3(cluster_magenta(:,1),cluster_magenta(:,2),cluster_magenta(:,3),'magenta')
    scatter3(cluster_yellow(:,1),cluster_yellow(:,2),cluster_yellow(:,3),'yellow')
    scatter3(cluster_cyan(:,1),cluster_cyan(:,2),cluster_cyan(:,3),'cyan')
    title('K-Means');xlabel('x'); ylabel('y'); zlabel('z');
end
