import random

# vertices = 50 thousand (20000000)
# edges = 700 million (300000000)

vertices_num = 100000
edges_num = 5000000

curr = 0
j = 1
print("Writing Edges")
print()

for j in range(1,25):
	f = open(f"graph_edges_{j}.txt","w")
	for i in range(edges_num):
		f.write(str(random.randint(0,vertices_num-1)))
		f.write(" ")
		f.write(str(random.randint(0,vertices_num-1)))
		f.write('\n')
	f.close()
	print(f"Iter {j} done")
	j += 1

print()
f = open(f"graph_vertices.txt","w")
print("Writing Vertices")

for i in range(vertices_num):
	f.write(str(i))
	f.write('\n')


