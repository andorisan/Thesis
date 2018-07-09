function frames = get_frames(tensor, frames_type, noise)

    if frames_type == 'simulated'  %________________________________________
        % Extract 30x30 frame that contains no missing data
        % and use it to create a tensor of 30 identical frames
        
        % Fix random seed
        rng(7);
        
        frames = [];
        for i=1:30
            frames(:,:,i) = tensor(491:520,531:560,23);
        end 

        %Add normally distributed noise to each pixel 
        for i=1:length(frames(:,1,1))
             for j=1:length(frames(1,:,1))
                 for k=1:length(frames(1,1,:))
                     if noise == 'no_noise_'
                        frames(i,j,k) = frames(i,j,k) + normrnd(0,0.00000000001);
                     end
                     if noise == 'add_noise'
                        frames(i,j,k) = frames(i,j,k) + normrnd(0,0.02);
                     end 
                 end
             end
         end
    end%___________________________________________________________________


    if frames_type == 'real_data'%________________________________________
        % Real data: Find frames in dataset that contain no missing data
        frames_with_no_nan = [];

        for i = 1:length(tensor(1,1,:))
            if sum(sum(isnan(tensor(491:520,531:560,i)))) == 0
                frames_with_no_nan = [frames_with_no_nan; i];
                %display(i);
            end
        end
        
        frames = tensor(491:520,531:560,frames_with_no_nan);
    end%___________________________________________________________________

end 