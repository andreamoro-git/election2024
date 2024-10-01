#! /usr/bin/env python3
#%% 
# get data of polls by state from url
# and simulate election outconmes from the electoral college
# and write the results to a file
import pandas as pd
import numpy as np
import random
from states import states, votes, stcode
import math
from datetime import datetime

#%% get the polls data

url = 'https://static.dwcdn.net/data/fgLCD.csv'
url = 'https://static.dwcdn.net/data/dFSDp.csv'
#url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'
url = 'https://www.electoral-vote.com/evp2024/Pres/pres_polls.csv'

from urllib.request import Request, urlopen
req = Request(url, headers= {"User-Agent" : "Mozilla/5.0"})

polls  = pd.read_csv(urlopen(req))

#%%


# Set up some parameters
debug = 1
pollspagefile = 'pollspage.html'
numsamples = 20000
htmlfile = 'try.html'
emailfile = 'results.txt'
statsfile = 'stats.raw'
datadir = "data/"
figdir = "../assets/"
savedate = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Initialize some variables
normupbound = 5
normstep = 0.01
random.seed(1234)

# Produce an array with the normal cdf
def normal_cdf(upbound, step):
    value = [-upbound + step * i for i in range(int(2 * upbound / step) + 1)]
    pdf = [(1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((v - step / 2) ** 2)) for v in value]
    cdf = [sum(pdf[:i + 1]) for i in range(len(pdf))]
    cdf = [c / cdf[-1] for c in cdf]
    return cdf

cdf = normal_cdf(normupbound, normstep)

# Get vote shares from polls
rep_votes1 = 0
tie_votes1 = 0

poll = {}  # Assuming poll data is available

for state in states:
    poll[state] = {}
    # skip if DC
    if state == 'District of Columbia':
        poll[state]['GOP'] = 0
        poll[state]['share'] = 0
        poll[state]['moe'] = 3
        continue
    poll[state]['GOP'] = polls[polls['State'] == state].iloc[0]['GOP']
    poll[state]['Dem'] = polls[polls['State'] == state].iloc[0]['Dem']
    poll[state]['moe'] = 3
    poll[state]['share'] = poll[state]['GOP'] / (poll[state]['GOP'] + poll[state]['Dem'])


#%%
# now simulate outcomes 
states = list(poll.keys())

# Vectorized alternative computation
dem_votes_all = []
repwin = 0
tie = 0

# Generate random draws for all states and samples at once
means = np.array([poll[state]['share'] for state in states])
std_devs = np.array([poll[state]['moe'] / 200 for state in states])
draws = np.random.normal(means, std_devs, (numsamples, len(states)))

# Determine which states are won by Republicans for each sample
rep_wins = draws > 0.5

# Calculate Republican votes for each sample
rep_votes_all = np.sum(rep_wins * np.array([votes[state] for state in states]), axis=1)

# Calculate Democratic votes for each sample
dem_votes_all = 538 - rep_votes_all

# Calculate the number of Republican wins and ties
repwin = np.sum(rep_votes_all > 269)
tie = np.sum(rep_votes_all == 269)

#%%
# Compute proportion of wins for republicans, ties, and democrats
propb = {}
propt = {}
propk = {}


proprep = repwin / numsamples * 100
proptie = tie / numsamples * 100
propdem = 100 - proprep - proptie

print(f"{proprep} {proptie}")

avgdemvotes = sum(dem_votes_all) / numsamples

#%% 
# plot the distribution of electoral votes
# without using seaborn
import matplotlib.pyplot as plt
import numpy as np

# Create a figure and a set of subplots
fig, ax = plt.subplots( figsize=(8, 4))
ax.set_title('Distribution of Harris Electoral Votes (average: ' + str(round(avgdemvotes, 1)) + ')')
ax.set_xlabel('Electoral Votes')
ax.set_ylabel('Frequency')
ax.set_ylim

#remove axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

bins = np.append(np.arange(219,270,10),np.arange(270,380,10))

# compute frequencies'
freq = []
freq.append(np.sum(dem_votes_all < bins[0]))
for i in range(1,len(bins)-1):
    freq.append(np.sum((dem_votes_all >= bins[i]) & (dem_votes_all < bins[i+1]) ))
freq = np.array(freq)/numsamples

colors = ['red' if x < 269 else 'blue' for x in bins[:-1]]
colors[np.where(bins==269)[0][0]]= 'green'

a = list(bins[1:-1])
a.remove(270)

ax.set_xticks(a)
ax.set_xticklabels(a)

replabel = 'Trump Wins'+ ' (' + str(round(proprep,1)) + '%)'
tielabel = 'Tie'+ ' (' + str(round(proptie,1)) + '%)'
demlabel = 'Harris Wins'+ ' (' + str(round(propdem,1)) + '%)'
handles = {}
for i in range(len(bins) - 1):
    [print(bins[i], freq[i])]
    if bins[i] < 269:
        handles[bins[i]] = ax.bar(bins[i]+5, freq[i], color=colors[i], width=9, label=replabel)
    elif bins[i] == 269:
        handles[bins[i]] = ax.bar(bins[i]+.5, freq[i], color=colors[i], width=1, label=tielabel)
    else:
        handles[bins[i]] = ax.bar(bins[i]+5, freq[i], color=colors[i], width=9, label=demlabel)


ax.legend(handles= [ handles[bins[0]],handles[269],handles[bins[-2]]])
fig.savefig(figdir+'harrisvotesdist.png', bbox_inches='tight')


# %% 
# create a dataframe with date and harris expected votes,
# and write to a file
try: 
    df = pd.read_csv(datadir+'harrisvotes.csv')
    if savedate not in df['date'].values:
        df = df.append({'date': savedate, 'votes': avgdemvotes, 'demprob' : propdem, 'tieprob': proptie}, ignore_index=True)
except:
    df = pd.DataFrame({'date': [savedate], 'votes': [avgdemvotes]})

df.to_csv( datadir+'harrisvotes.csv', index=False)

# %%
# plot a figure of harrisvotes by date from the data frame
fig, (ax,ax2) = plt.subplots(1,2, figsize=(8,3))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# substring the date ecolumn to get the month and day
df['dm'] = df['date'].str[5:10]
ax.plot(df['dm'], df['votes'], label='Harris Expected Votes', color='blue')
ax.set_title('Dem expected electoral votes')
ax2.set_title('Dem win frequencies')
ax.set_xlabel('Date')
ax.set_ylabel('Electoral Votes')
ax.set_ylim(270-70, 270+70)
ax.set_yticks(np.arange(210, 331, 20))
ax2.set_ylim(0, 100)
ax2.plot(df['dm'], df['demprob'], color='blue', linestyle=':', label='Harris Win Probability (right)')

# horizontal line at 270
ax.axhline(y=270, color='r', linestyle='--', xmin=0.04, xmax=0.97)
ax2.axhline(y=50, color='r', linestyle='--', xmin=0.04, xmax=0.97)

# limit the width of the axhline within the same range of the plot lines
plt.xticks(rotation=45)
plt.show()

# save figure
fig.savefig(figdir+'harrisvotes.png', bbox_inches='tight')
# %%
