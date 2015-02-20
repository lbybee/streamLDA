from streamlda import StreamLDA
from util import print_topics
import MySQLdb
import datetime
import numpy as n
from pylab import *
import random
import cPickle
import time


""" This script loads a MySQL database of tweets and extracts the
tweets for a subset of dates and updates the online LDA model with
them"""


def loadNTweets(host, user, passwd, db, table, date):
    """extracts the list of tweet text and user ids that meet the
    date criteria"""

    tweets = []

    rdb = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db, use_unicode=True)
    cursor = rdb.cursor()

    cursor.execute('SET NAMES utf8mb4')
    cursor.execute("SET CHARACTER SET utf8mb4")
    cursor.execute("SET character_set_connection=utf8mb4")

    search_str = "SELECT UID TEXT FROM %s WHERE YEAR > %d AND MONTH > %d AND DAY > %d AND HOUR > %d" % (table, date.year, date.month, date.day, date.hour)
    cursor.execute(search_str)
    data = cursor.fetchall()
    return data


def processTweets(tweets):
    """takes in a list of tweets and converts into a tuple of user tweet
    pairs, where each pair contains the user name and the full text for
    all the users tweets in the last period, it also makes the tweets
    usable by oLDA"""

    c_tweets = {}
    for t in tweets:
        u_id = t[0]
        if u_id not in c_tweets:
            c_tweets[u_id] = ""
        c_tweets[u_id] += t[1].encode("utf-8")

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


def fullRun(num_topics, alpha, eta, tau0, kappa, date, s_count,
            host, user, passwd, db, table, olda=None):
    """keeps the lights on"""

    # first we initalize the olda mod as well as inital run time
    t_main = datetime.datetime.now()
    if olda is None:
        olda = StreamLDA(num_topics, alpha, eta, tau0, kappa, sanity_check=False)
    print "oLDA initalized"
    i = 0
    print "beginning model updating..."

    while True:
        
        tweets = loadNTweets(host, user, passwd, db, table, date)
        c_ids, c_tweets = processTweets(tweets)

        # store the ids so that they can be linked up with the documents later
        writeUserNames(c_ids, "u_ids.txt")
    
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

