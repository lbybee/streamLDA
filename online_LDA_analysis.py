from scipy.stats.stats import pearsonr 
import itertools


def loadGamma(f_list):
    """takes in a list of gamma files and puts them all into
    one series"""

    topic_list = []
    for f in f_list:
        data = open(f, "rb")
        content = data.read().split("\n")[:-1]
        topic_list.extend([r.split(" ") for r in content])
        data.close()
    return topic_list


def loadNodes(node_f):
    """loads a node file that contains each node along with
    the period that they correspond to"""

    node_data = open(node_f, "rb")
    node_content = node_data.read().split("\n")[:-1]
    return [r.split(" ") for r in node_content]


# note that the result from loadGamma and loadNodes should be the same
# length


def joinDataDict(gamma_list, node_list):
    """takes in a gamma list and node list produced by the above code
    and converts this into a dictionary"""

    # first we need to get the maximum number of periods
    dates = [r[1] for r in node_list]
    mx_date = max(dates)

    # we need to get the number of topics.
    K = len(gamma_list[0])

    data_dict = {}

    # now we iterate through the paired lists and build the data dict
    for g, n in zip(gamma_list, node_list):
        if n[0] not in data_dict:
            data_dict[n[0]] = {}
        data_dict[n[0]][n[1]] = g
    
    # now we need to add in any missing days, we just set these to 0
    # to correspond to no signal.
    for k in data_dict:
        for d in 0:mx_date:
            if d not in data_dict[k]:
                data_dict[k][d] = [0] * K

    return data_dict


def genNodeCorrelationDict(data_dict, s_date, e_date, period_mn, period_mx,
                           mn_date, mx_date):
    """whats this function does is take in the data dictionary and
    for each node it calculates the correlation between that nodes
    topic proportions for the specified period starting at s_date"""

    corr_dict = {}
    
    # get node pairs
    n_pairs = itertools.combinations(data_dict.keys(), 2)

    for pair in n_pairs:
        corr_dict[pair] = []
        for s in range(s_date, e_date):
            s_range = range(min((mn_date, s_date - period_mn)), min((mx_date, s_date + period_mx)))
            val_i = []
            val_j = []
            for d in s_range:
                val_i.extend(data_dict[pair[0]][d])
                val_j.extend(data_dict[pair[1]][d])
            corr_dict[pair].append(personr(val_i, val_j))

    return corr_dict


def genTopicCorrelationList(data_dict, s_date, e_date, period_mn, period_mx,
                            mn_date, mx_date, K):
    """this function takes in the data dictionary and generates a correlation
    matrix between all pairs of topics for a number of rolling windows, this
    really needs to be moved over to numpy, I'm rusty on some of this so this
    is a temporary fix"""

    corr_list = []

    t_pairs = itertools.combinations(range(1, K+1), 2)

    for s in range(s_date, e_date):
        pair_list = []
        s_range = range(min((mn_date, s_date - period_mn)), min((mx_date, s_date + period_mx)))
        temp_list = []
        for d in data_dict:
            for k in s_range:
                temp_list.append(data_dict[d][k])
        for p in t_pairs:
            pair_list.append(pearsonr([r[p[0]] for r in temp_list], [r[p[1]] for r in temp_list]))
        corr_list.append(pair_list)

    return corr_list


def findSignificantNodeTopic(data_dict, threshold, node_n, n_value, topic_n, s_date, e_date,
                             period_mn, period_mx, mn_date, mx_date):
    """determines whether the n_value is significantly different from the distribution
    of topics that we have seen before"""


