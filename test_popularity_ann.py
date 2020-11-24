from popularity_ann import PopularityANN, generate_sparse_connection_matrix



f = PopularityANN(N=400, c=50)
print(f.connections)
print(f.C)