f = open('test3.txt','r')
f2 = open('test3g.txt','w')

l = f.readlines()
l = [x.strip() for x in l]
g = ['<Comedy>', '<Action>', '<Adventure>', '<Crime>', '<Drama>', '<Fantasy>', '<Horror>', '<Music>', '<Romance>', '<Sci-Fi>', '<Thriller>']

for x in l:
    for y in g:
        f2.write(y+" "+x+"\n")