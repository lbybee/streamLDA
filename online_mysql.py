from streamlda import StreamLDA
from dirichlet_words import DirichletWords
from nltk import FreqDist
from util import print_topics
import MySQLdb
import datetime
from datetime import timedelta
import numpy as n
from pylab import *
import random
import cPickle
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


def crashPrep(olda_mod, pickle_f):
    """stores the olda mod in case of a crash"""

    pickle_data = open(pickle_f, "wb")
    out_data = olda_mod.__dict__
    out_data["_lambda"] = out_data["_lambda"].__dict__
    out_data["_lambda"]["_report"] = None
    out_data["_lambda"]["_words"] = out_data["_lambda"]["_words"].__dict__
    out_data["_lambda"]["_topics"] = [t.__dict__ for t in out_data["_lambda"]["_topics"]]
    cPickle.dump(out_data, pickle_data)
    pickle_data.close()


def loadCrashedRes(olda_f):
    """loads the results in case of a crash, this includes the olda obj
    and the crash date"""

    pickle_data = open(olda_f, "rb")
    data = cPickle.load(pickle_data)
    pickle_data.close()
    t_lambda_dict = data["_lambda"]

    t_words = FreqDist()
    t_words.__dict__ = data["_lambda"]["_words"]

    t_topics = [FreqDist() for t in data["_lambda"]["_topics"]]
    for i in range(len(data["_lambda"]["_topics"])):
        t_topics[i].__dict__ = data["_lambda"]["_topics"][i]

    t_lambda_dict["_words"] = t_words
    t_lambda_dict["_topics"] = t_topics

    t_lambda = DirichletWords()
    t_lambda.__dict__ = t_lambda_dict

    data["_lambda"] = t_lambda
    
    olda_mod = StreamLDA()
    olda_mod.__dict__ = data

    return olda_mod


def writeUserNames(c_names, f_name):
    """appends the user names to the file"""

    f_data = open(f_name, "ab")
    for n in c_names:
        f_data.write("%s\n" % n)
    f_data.close()


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days) + 1):
        yield start_date + timedelta(n) 
