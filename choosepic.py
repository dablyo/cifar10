import os
import sys
import os.path
import random

def choose(sourcedir,destdir,num):
    if not os.path.exists(sourcedir):
        print "{} doesn't exist.".format(sourcedir)
        exit()
    if not os.path.exists(destdir):
        print "{} doesn't exist.".format(destdir)
        exit()

    res=set()
    for root,dirnames,filenames in os.walk(sourcedir):
        if (len(filenames)<num):
            print "amount: {}".format(len(filenames))
            print "amount of files is smaller or equal {}, exit".format(num)
            exit()
        while(len(res)<num):
            thefile=random.choice(filenames)
            sourcefile=os.path.join(sourcedir,thefile)
            destfile=os.path.join(destdir,thefile)
            cmd='cp {}  {}'.format(sourcefile,destfile)
            res.add(cmd)

    x=0
    for i in res:
        os.system(i)
        x+=1
        print '{} {}'.format(x,i)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage:"
        print "python choose.py $source $destination $number"
        exit()
    choose(sys.argv[1],sys.argv[2],int(sys.argv[3]))

