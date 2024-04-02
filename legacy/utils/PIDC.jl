using NetworkInference

path = "Benchmark/Trees10"

for r = 1:10
	data = string(path, "/Data/data_$(C)_10_12_$(r).txt")
	score = string(path, "/PIDC/score_$(C)_10_12_$(r).txt")
	infer_network(data, PIDCNetworkInference(), out_file_path=score)
end
