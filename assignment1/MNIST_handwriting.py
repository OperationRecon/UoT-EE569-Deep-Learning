from sklearn import datasets
mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target'].astype(int)
