function convert(h5_file)

outfpath = [h5_file '.mat'];

TOTAL_ACT_IDS = 983;  % = max([RELEASE.act.act_id])
DATA_DIR='../src/data/mpii/mpii_tfrecords';

sample_ids = dlmread(fullfile(DATA_DIR, 'test_ids.txt'));
% scores = zeros(numel(sample_ids), numel(class_ids));
class_ids = {};

cid = 0;
fid = fopen(fullfile(DATA_DIR, 'classes.txt'), 'r');
while ~feof(fid)
  line = fgetl(fid);
  cid = cid + 1;
  parts = strsplit(line, ';');
  class_ids{cid} = parts{1};
  % nums = cellfun(@str2num, strsplit(parts{2}, ','));
  % cls_to_ids{cid} = nums;
end

scores = h5read(h5_file, '/logits')';
% for cid = 1 : numel(cls_to_ids)
%   targets = cls_to_ids{cid};
%   for i = 1 : numel(targets)
%     scores(:, targets(i)) = logits(:, cid);
%   end
% end

save(outfpath, 'sample_ids', 'class_ids', 'scores');
