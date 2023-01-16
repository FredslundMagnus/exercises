import os, sys, tarfile

tar = tarfile.open(sys.argv[1] + '.tgz', 'r:gz')
for item in tar:
    tar.extract(item)