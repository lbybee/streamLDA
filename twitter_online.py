from pymongo import MongoClient
from streamlda import StreamLDA
from util import print_topics
import datetime
import numpy as n
from pylab import *
import random
import cPickle
import time


""" This script loads a mongodb database of tweets scraped from the Twitter
API and builds a topic model as new Tweets come in"""


def loadNTweets(db_n, col_n, date, e_date):
    """loads a list of Tweets that aren't in the model yet"""

    client = MongoClient()
    db = client[db_n]
    col = db[col_n]

    tweets = []
    i = 0

    for t in col.find({"created_at": {"$gt": date, "$lt": e_date}}):
        tweets.append(t)
        print i, "extracting tweets after", date
        i += 1

    return tweets


def processTweets(tweets):
    """takes in a list of tweets and converts into a tuple of user tweet
    pairs, where each pair contains the user name and the full text for
    all the users tweets in the last period, it also makes the tweets
    usable by oLDA"""

    c_tweets = {}
    for t in tweets:
        u_name = t["user"]["screen_name"]
        if u_name not in c_tweets:
            c_tweets[u_name] = ""
        c_tweets[u_name] += t["text"].encode("utf-8")

    return c_tweets.keys(), c_tweets.values()


def writeModelRes(olda_mod, gamma, date):
    """takes in an stream LDA (olda so as not to confuse with supervised LDA)
    and writes the values to txt files for later analysis"""

    str_date = date.strftime("%Y-%m-%d-%H-%M-%S")
    n.savetxt("lambda-%s.txt" % str_date, olda_mod._lambda.as_matrix())
    n.savetxt("gamma-%s.txt" % str_date, gamma)


def crashPrep(olda_mod, date, pickle_f):
    """stores the olda mod in case of a crash"""

    pickle_data = open(pickle_f, "wb")
    cPickle.dump({"date": date, "mod": olda_mod}, pickle_data)
    pickl_data.close()


def loadCrashedRes(olda_f):
    """loads the results in case of a crash, this includes the olda obj
    and the crash date"""

    pickle_data = open(olda_f, "rb")
    data = cPickle.load(pickle_data)
    pickle_data.close()
    return data


def writeUserNames(c_names, f_name):
    """appends the user names to the file"""

    f_data = open(f_name, "ab")
    for n in c_names:
        f_data.write("%s\n" % n)
    f_data.close()


def fullRun(db_n, col_n, num_topics, alpha, eta, tau0, kappa, date, s_count,
            olda=None):
    """keeps the lights on"""

    # first we initalize the olda mod as well as inital run time
    t_main = datetime.datetime.now()
    if olda is None:
        olda = StreamLDA(num_topics, alpha, eta, tau0, kappa, sanity_check=False)
    print "oLDA initalized"
    i = 0

    # give the db enough time so that we aren't pulling tweets that are just
    # coming in, this can crash the db
    time.sleep(30)

    while True:
        
        tweets = loadNTweets(db_n, col_n, date, t_main)
        c_names, c_tweets = processTweets(tweets)

        # store the ids so that they can be linked up with the documents later
        writeUserNames(c_names, "u_names.txt")
    
        # we initalize an iteration specific run time so that when we
        # grab new data we are only grabbing data we haven't added before
        t_run = datetime.datetime.now()
        date = t_run

        # now we update the olda obj
        gamma, bound = olda.update_lambda(c_tweets)
        wordids, wordcts = olda.parse_new_docs(c_tweets)

        # write each iteration to their files
        writeModelRes(olda, gamma, i)

        # now store the model in case of a crash
        crashPrep(olda, t_run, "backup.pkl")

        print i, datetime.datetime.now() - t_run, datetime.datetime.now() - t_main
        time.sleep(s_count)
