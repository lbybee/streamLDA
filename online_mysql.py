from streamlda import StreamLDA
from util import print_topics
import MySQLdb
import datetime
from datetime import timedelta
import numpy as n
from pylab import *
import random
import dill 
import time


""" This script loads a MySQL database of tweets and reddit comments
and extracts the tweets and comments for a subset of dates and updates
the online LDA model with them"""


def loadNTweets(host, user, passwd, db, table, s_date, e_date):
    """extracts the list of tweet text and user ids that meet the
    date criteria"""

    tweets = []

    rdb = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db, use_unicode=True)
    cursor = rdb.cursor()

    cursor.execute('SET NAMES utf8mb4')
    cursor.execute("SET CHARACTER SET utf8mb4")
    cursor.execute("SET character_set_connection=utf8mb4")

    search_str = "SELECT UID, TEXT FROM %s WHERE YEAR >= %d AND MONTH >= %d AND DAY >= %d AND HOUR >= %d AND YEAR <= %d AND MONTH <= %d AND DAY <= %d AND HOUR <= %d " % (table, s_date.year, s_date.month, s_date.day, s_date.hour, e_date.year, e_date.month, e_date.day, e_date.hour)
    cursor.execute(search_str)
    data = cursor.fetchall()
    data = [r for r in data]
    return data


def loadNComments(host, user, passwd, db, table, s_date, e_date):
    """extracts the list of comment text and subreddit ids that meet the
    date criteria"""

    tweets = []

    rdb = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db, use_unicode=True)
    cursor = rdb.cursor()

    cursor.execute('SET NAMES utf8mb4')
    cursor.execute("SET CHARACTER SET utf8mb4")
    cursor.execute("SET character_set_connection=utf8mb4")

    search_str = "SELECT SUBREDDIT, BODY FROM %s WHERE YEAR >= %d AND MONTH >= %d AND DAY >= %d AND HOUR >= %d AND YEAR <= %d AND MONTH <= %d AND DAY <= %d AND HOUR <= %d " % (table, s_date.year, s_date.month, s_date.day, s_date.hour, e_date.year, e_date.month, e_date.day, e_date.hour)
    cursor.execute(search_str)
    data = cursor.fetchall()
    data = [r for r in data]
    return data


def processSourceTextPair(text_pairs, user_ids=[]):
    """takes in a list of source text pairs and converts into a tuple 
    where each pair contains the source name/id and the full text for
    all the source text pairs in the last period, it also makes the data
    usable by oLDA"""

    if len(user_ids) == 0:
        c_data = {}
        for t in text_pairs:
            s_id = t[0]
            if s_id not in c_data:
                c_data[s_id] = ""
            c_data[s_id] += t[1].encode("utf-8")
    else:
        c_data = {k : "" for k in user_ids}
        for t in text_pairs:
            s_id = t[0]
            if s_id in c_data:
                c_data[s_id] += t[1].encode("utf-8")
        c_data = {k : c_data[k] for k in c_data if c_data[k] != ""}
    return c_data.keys(), c_data.values()


def writeModelRes(olda_mod, gamma, date):
    """takes in an stream LDA (olda so as not to confuse with supervised LDA)
    and writes the values to txt files for later analysis"""

    str_date = date.strftime("%Y-%m-%d-%H-%M-%S")
    n.savetxt("lambda-%s.txt" % str_date, olda_mod._lambda.as_matrix())
    n.savetxt("gamma-%s.txt" % str_date, gamma)


def crashPrep(olda_mod, dill_f):
    """stores the olda mod in case of a crash"""

    dill_data = open(dill_f, "wb")
    out_data = {}
    out_data["_K"] = olda_mod._K
    out_data["_alpha"] = olda_mod._alpha
    out_data["_eta"] = olda_mod._eta
    out_data["_tau0"] = olda_mod._tau0
    out_data["_kappa"] = olda_mod._kappa
    out_data["sanity_check"] = olda_mod.sanity_check
    out_data["_D"] = olda_mod._D
    out_data["_batches_to_date"] = olda_mod._batches_to_date
    out_data["recentbatch"] = olda_mod.recentbatch
    out_data["_lambda"] = olda_mod._lambda
    out_data["_lambda_mat"] = olda_mod._lambda_mat
    out_data["_Elogbeta"] = olda_mod._Elogbeta
    out_data["_expElogbeta"] = olda_mod._expElogbeta

    dill.dump(out_data, dill_data)
    dill_data.close()


def loadCrashedRes(olda_f):
    """loads the results in case of a crash, this includes the olda obj
    and the crash date"""

    dill_data = open(olda_f, "rb")
    data = dill.load(dill_data)
    dill_data.close()
    return data


def writeUserNames(c_names, f_name):
    """appends the user names to the file"""

    f_data = open(f_name, "ab")
    for n in c_names:
        f_data.write("%s\n" % n)
    f_data.close()


def modelDumpTest(num_topics, alpha, eta, tau0, kappa, date, s_count,
                  host, user, passwd, db, table, source="twitter",
                  user_f="user_ids.pkl", olda=None, in_data_f=""):
    """tests the output"""

    # first we initalize the olda mod as well as inital run time
    t_main = datetime.datetime.now()
    if olda is None:
        if in_data_f == "":
            olda = StreamLDA(num_topics, alpha, eta, tau0, kappa)
        else:
            in_data = loadCrashedRes(in_data_f)
            olda = StreamLDA(num_topics, alpha, eta, tau0, kappa, prev_model=in_data)
    print "oLDA initalized"
    i = 0
    print "beginning model updating..."

    # we initalize an iteration specific run time so that when we
    # grab new data we are only grabbing data we haven't added before
    t_run = datetime.datetime.now()
        
    if source == "twitter":
        user_data = open(user_f, "rb")
        user_ids = dill.load(user_data)
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
    return olda


def fullRun(num_topics, alpha, eta, tau0, kappa, date, s_count,
            host, user, passwd, db, table, source="twitter",
            user_f="user_ids.pkl", olda=None, in_data_f=""):
    """keeps the lights on, some good starting values
    alpha = 50/num_topics, eta = 0.1, tau0=1 kappa=0.7"""

    # first we initalize the olda mod as well as inital run time
    t_main = datetime.datetime.now()
    if olda is None:
        if in_data_f == "":
            olda = StreamLDA(num_topics, alpha, eta, tau0, kappa)
        else:
            in_data = loadCrashedRes(in_data_f)
            olda = StreamLDA(num_topics, alpha, eta, tau0, kappa, prev_model=in_data)
    print "oLDA initalized"
    i = 0
    print "beginning model updating..."

    while True:

        # we initalize an iteration specific run time so that when we
        # grab new data we are only grabbing data we haven't added before
        t_run = datetime.datetime.now()
        
        if source == "twitter":
            user_data = open(user_f, "rb")
            user_ids = dill.load(user_data)
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


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days) + 1):
        yield start_date + timedelta(n) 


def modelInit(num_topics, alpha, eta, tau0, kappa, s_date, e_date,
              host, user, passwd, db, table, source="twitter",
              user_f="user_ids.pkl", olda=None):
    """initalizes the model for a series of days"""

    # first we initalize the olda mod as well as inital run time
    t_main = datetime.datetime.now()
    if olda is None:
        olda = StreamLDA(num_topics, alpha, eta, tau0, kappa)
    print "oLDA initalized"
    i = 0
    print "beginning model updating..."

    for d in daterange(s_date, e_date):
        
        if source == "twitter":
            user_data = open(user_f, "rb")
            user_ids = dill.load(user_data)
            user_data.close()
            tweets = loadNTweets(host, user, passwd, db, table, d, d + timedelta(1))
            c_ids, c_text = processSourceTextPair(tweets, user_ids)
        else:
            comments = loadNComments(host, user, passwd, db, table, d, d + timedelta(1))
            c_ids, c_text = processSourceTextPair(comments)


        if len(c_text) > 0:
            # store the ids so that they can be linked up with the documents later
            writeUserNames(c_ids, source + "_u_ids.txt")
    
            # we initalize an iteration specific run time so that when we
            # grab new data we are only grabbing data we haven't added before
            t_run = datetime.datetime.now()
            date = t_run

            # now we update the olda obj
            gamma, bound = olda.update_lambda(c_text)
            wordids, wordcts = olda.parse_new_docs(c_text)


            # write each iteration to their files
            writeModelRes(olda, gamma, d)
            print "models written"

            # now store the model in case of a crash
            crashPrep(olda, source + "_backup.pkl")

            print i, datetime.datetime.now() - t_run, datetime.datetime.now() - t_main
        else:
            print i, datetime.datetime.now() - t_run, datetime.datetime.now() - t_main, "No data for this period"
        i += 1
