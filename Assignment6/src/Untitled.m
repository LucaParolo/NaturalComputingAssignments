addpath src/adrem
addpath src/evaluation

feature = 'vgg';
percentage = 60;

data = load_dataset('office-caltech', feature);

[x_src, x_tgt] = preprocess(data.x{1}, data.y{1}, data.x{2}, 'joint-std');

% Find how many rows we should remove
num_removed_rows = int16((size(x_tgt,1)/100)*percentage);
% Randomly select indices 
removed_idx = randperm(size(x_tgt,1), num_removed_rows);

% Add the selected portion from the target to the source
x_src = [x_src; x_tgt(removed_idx,:)];
y_src = [data.y{1}; data.y{2}(removed_idx,:)];

% Remove the selected portion from the target
[x_tgt,~] = removerows(x_tgt,'ind',removed_idx);
[y_tgt,~] = removerows(data.y{2},'ind',removed_idx);

% Predict
y = predict_adrem(x_src, y_src, x_tgt);
mean(y == y_tgt)