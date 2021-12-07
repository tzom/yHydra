def batched_list(x, batch_size):
	for i in range(0, len(x), batch_size):
		yield x[i:i + batch_size]

def unbatched_list(x):
  return [item for sublist in x for item in sublist]