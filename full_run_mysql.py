from streamlda import StreamLDA
from online_mysql import *
from util import print_topics
import MySQLdb
import datetime
from datetime import timedelta
import numpy as n
from pylab import *
import random
import cPickle
import dill 
import time
import os
import sys


num_topics = int(sys.argv[1])
alpha = float(sys.argv[2])
eta = float(sys.argv[3])
tau0 = float(sys.argv[4])
kappa = float(sys.argv[5])
date = datetime.datetime(int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), 0, 0)
s_count = int(sys.argv[10])
host = sys.argv[11]
user = sys.argv[12]
passwd = sys.argv[13]
db = sys.argv[14]
table = sys.argv[15]
source = sys.argv[16]
user_f = sys.argv[17]
load_prev = int(sys.argv[18])


os.chdir(sys.argv[19])

# first we initalize the olda mod as well as inital run time
t_main = datetime.datetime.now()
if load_prev:
    olda = loadCrashedRes(source + "_backup.pkl")
else:
    olda = StreamLDA(num_topics, alpha, eta, tau0, kappa)
print "oLDA initalized"
i = 0
print "beginning model updating..."

while True:

    # we initalize an iteration specific run time so that when we
    # grab new data we are only grabbing data we haven't added before
    t_run = datetime.datetime.now()
     
    if source == "twitter":
        user_data = open(user_f, "rb")
        user_ids = cPickle.load(user_data)
        user_data.close()
        tweets = loadNTweets(host, user, passwd, db, table, date, t_run)
        c_ids, c_text = processSourceTextPair(tweets, user_ids)
    else:
        comments = loadNComments(host, user, passwd, db, table, date, t_run)
        c_ids, c_text = processSourceTextPair(comments)

    # store the ids so that they can be linked up with the documents later
    writeUserNames(c_ids, source + "_u_ids.txt")
   
    date = t_run

    # now we update the olda obj
    gamma, bound = olda.update_lambda(c_text)
    wordids, wordcts = olda.parse_new_docs(c_text)


    # write each iteration to their files
    writeModelRes(olda, gamma, date)
    print "models written"

    # now store the model in case of a crash
    crashPrep(olda, source + "_backup.pkl")

    print i, datetime.datetime.now() - t_run, datetime.datetime.now() - t_main
    i += 1
    time.sleep(s_count)
