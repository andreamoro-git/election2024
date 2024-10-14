#! /usr/bin/env python3
#%% 
# simulate 2024 election outcomes
# get data of polls by state from url
# and simulate election outconmes from the electoral college
# and write the results to a file
import pandas as pd
import numpy as np
import random
from states import votes, nsurls
from datetime import datetime
from urllib.request import Request, urlopen

# make sure I can debug
import os
if os.getcwd()[-4:] == 'code':
    os.chdir('..')

# Set up some parameters
moe_if_missing = 3
debug = 1
pollspagefile = 'pollspage.html'
numsamples = 20000
htmlfile = 'try.html'
emailfile = 'results.txt'
statsfile = 'stats.raw'
datadir = "_code/data/"
figdir = "assets/"
savedate = datetime.now().strftime("%Y-%m-%d")

random.seed(1234)
fullnames = {'ev': 'electoral-vote', 'ns': 'natesilver'}
states = votes.keys()



#%% 
# get the polls data

# start with the electoral-vote data
url_electoralvote = 'https://www.electoral-vote.com/evp2024/Pres/pres_polls.csv'

req = Request(url_electoralvote, headers= {"User-Agent" : "Mozilla/5.0"})
polls  = pd.read_csv(urlopen(req))
polls['source'] = 'ev'

# reformat month from Apr to 04
polls['Date'] = polls['Date'].apply(lambda x: datetime.strptime(x, '%b %d').strftime('2024-%m-%d'))

# continue with natesilver data  state by state
for key in nsurls.keys():
    req = Request(nsurls[key], headers= {"User-Agent" : "Mozilla/5.0"})
    polls_ns = pd.read_csv(urlopen(req))
    polls_ns = polls_ns.rename(columns={'state':'State', 'trump':'GOP', 'harris':'Dem', 'modeldate': 'Date' })
    polls_ns['source'] = 'ns'
    polls_ns['Date'] = polls_ns['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d'))

    # concat the dataframe to the  polls dataframe
    polls = pd.concat([polls,polls_ns], ignore_index=True)

polls = polls[['Date','State','Dem','GOP','source','Pollster','trump_poll','harris_poll']]
polls['dayssince'] = polls['Date'].apply(lambda x: (datetime.now()- datetime.strptime(x, '%Y-%m-%d')).days)

# sort polls by state and date
polls = polls.sort_values(by=['State','dayssince'], ascending=True)
#%% 
# do the average of last 4 polls for ev
n_polls_keep = 4
latestEV = polls[polls['source']=='ev'].groupby(['State']).head(n_polls_keep)

# keep only the average by state, add the numeric_only option
latestEV = latestEV[['Dem','GOP','State']].groupby(['State']).mean()

# remove index and make it a column
latestEV.reset_index(inplace=True)
latestEV['source'] = 'ev'
latestEV = latestEV[['Dem','GOP','source','State']]

date = polls[polls['source']=='ev'].groupby(['State']).head(1)
date = date[['Date','dayssince','State']]

#merge dfs
latestEV = latestEV.merge(date, on='State')

#%%
# take the latest for ns
latestNS = polls[polls['source']=='ns'].groupby(['State']).head(1)

# keep only the average by state, add the numeric_only option
latestNS = latestNS[['Dem','GOP','State']].groupby(['State']).mean()
# remove index and make it a column
latestNS.reset_index(inplace=True)
latestNS['source'] = 'ns'
latestNS = latestNS[['Dem','GOP','source','State']]

date = polls[polls['source']=='ns'].groupby(['State']).head(1)
date = date[['Date','dayssince','State']]

#merge dfs
latestNS = latestNS.merge(date, on='State')

latest = pd.concat([latestEV, latestNS])

# add shareGOP and moe
latest['shareGOP'] = latest['GOP'] / (latest['Dem'] + latest['GOP'])
latest['moe'] = moe_if_missing

# convert long to wide
latest = latest.set_index(['source','State'])
latest = latest.unstack('source')
latest.columns = ['{}_{}'.format(var, source) for var, source in latest.columns]

# fill missing values
latest['shareGOP_ns']=latest['shareGOP_ns'].fillna(latest['shareGOP_ev'])
latest['moe_ns']=latest['moe_ns'].fillna(latest['moe_ev'])


# now simulate outcomes 

def simulate(source):
    dem_votes_all = []
    repwin = 0
    tie = 0
    
    shs = 'shareGOP_' + source
    mos = 'moe_' + source

    # Generate random draws for all states and samples at once
    means = latest[shs] #np.array([shares[source][state]['share'] for state in states])
    std_devs = latest[mos]/200 #np.array([shares[source][state]['moe'] / 200 for state in states])
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

    proprep = repwin / numsamples * 100
    proptie = tie / numsamples * 100
    propdem = 100 - proprep - proptie
    avgdemvotes = sum(dem_votes_all) / numsamples

    return {'demvotes':avgdemvotes,'demprob':propdem,'tieprob':proptie}, dem_votes_all

stats = {}
allsims = {}
stats['ev'], allsims['ev'] = simulate('ev')
stats['ns'], allsims['ns'] = simulate('ns')

for source in fullnames.keys():
    print(source, stats[source])
# %% 
# create a dataframe with date and harris expected votes,
# and write to a file
try: 
    df = pd.read_csv(datadir+'harrisvotes.csv')
    if savedate not in df['date'].values:
        if 'ev' not in df[df['date']==savedate]['source'].values:
            # use concat instead of append
            df = pd.concat([df, pd.DataFrame({'date':savedate,**stats['ev'],'source':'ev'}, index=[0])], ignore_index=True)
        if 'ns' not in df[df['date']==savedate]['source'].values:
            df = pd.concat([df, pd.DataFrame({'date':savedate,**stats['ns'],'source':'ns'}, index=[0])], ignore_index=True)
except:
    print('aa')
    df = pd.DataFrame({'date':savedate,**stats['ev'],'source':'ev'})
    df = pd.concat([df, pd.DataFrame({'date':savedate,**stats['ns'],'source':'ns'}, index=[0])], ignore_index=True)

df.to_csv( datadir+'harrisvotes.csv', index=False)

#%% 
# plot the distribution of electoral votes
import matplotlib.pyplot as plt
import numpy as np

# Create a figure and a set of subplots
fig, (ax) = plt.subplots(1,1, figsize=(8, 4))
ax.set_title('Distribution of Harris Electoral Votes')
ax.set_xlabel('Electoral Votes (10-votes bins)')
ax.set_ylabel('Frequency')

#remove axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
bins = np.append(np.arange(219,270,10),np.arange(270,370,10))

# compute frequencies'

freq = {}
for source in fullnames.keys():
    dem_votes_all = allsims[source]
    freq[source] = []
    freq[source].append(np.sum(dem_votes_all < bins[0]))
    for i in range(1,len(bins)-1):
        freq[source].append(np.sum((dem_votes_all >= bins[i]) & (dem_votes_all < bins[i+1]) ))
    freq[source] = np.array(freq[source])/numsamples

colors = ['red' if x < 269 else 'blue' for x in bins[:-1]]
colors[np.where(bins==269)[0][0]]= 'green'

a = list(bins[1:-1])
a.remove(270)

ax.set_xticks(a)
ax.set_xticklabels(a)
ax.set_yticks(np.arange(-.2,0.3,0.1))
ax.set_yticklabels(['.2','.1','0','.1','.2'])

replabel = {}
tielabel = {}
demlabel = {}
for source in fullnames.keys():
    repprob = 100 - stats[source]['demprob'] - stats[source]['tieprob']
    replabel[source] = 'Trump Wins'+ '(' + str(round(repprob,1)) + '%)'
    tielabel[source] = 'Tie'+ ' (' + str(round(stats[source]['tieprob'],1)) + '%)'
    demlabel[source] = 'Harris Wins'+ ' (' + str(round(stats[source]['demprob'],1)) + '%)'

handles = {}
handles2 = {}
for i in range(len(bins) - 1):
    if bins[i] < 269:
        handles[bins[i]] = ax.bar(bins[i]+5, freq['ev'][i], color=colors[i], width=9, label=replabel['ev'], alpha=1)
        # draw empty bars for natesilver
        handles2[bins[i]] = ax.bar(bins[i]+5, -freq['ns'][i], color=colors[i], width=9, label=replabel['ns'], hatch='.')
    elif bins[i] == 269:
        handles[bins[i]] = ax.bar(bins[i]+.5, freq['ev'][i], color=colors[i], width=1, label=tielabel['ev'], alpha=1)
        handles2[bins[i]] = ax.bar(bins[i]+.5, -freq['ns'][i], color=colors[i], width=1, label=tielabel['ns'],  hatch='.')
    else:
        handles[bins[i]] = ax.bar(bins[i]+5, freq['ev'][i], color=colors[i], width=9, label=demlabel['ev'], alpha=1)
        handles2[bins[i]] = ax.bar(bins[i]+5, -freq['ns'][i], color=colors[i], width=9, label=demlabel['ns'],  hatch='.')

#horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', xmin=0.04, xmax=0.97)
legend1 = ax.legend(handles= [ handles[bins[0]],handles[269],handles[bins[-2]]], title='electoral-vote '+'average: ' + str(round(stats['ev']['demvotes'], 1)))

# draw a second legend
legend2 = ax.legend(handles= [ handles2[bins[0]],handles2[269],handles2[bins[-2]]], loc='lower right', title='natesilver ''average: ' + str(round(stats['ns']['demvotes'], 1)))
ax.add_artist(legend1)
fig.savefig(figdir+'harrisvotesdist.png', bbox_inches='tight')


# %%
# plot a figure of harrisvotes by date from the data frame
fig, (ax,ax2) = plt.subplots(1,2, figsize=(8,3))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# substring the date ecolumn to get the month and day
df['dm'] = df['date'].str[5:10]
ax.plot(df[df['source']=='ev']['dm'], df[df['source']=='ev']['demvotes'], label='electoral-vote', color='blue')
ax.plot(df[df['source']=='ns']['dm'], df[df['source']=='ns']['demvotes'], ':', label='natesilver', color='blue')
ax.set_title('Harris expected electoral votes')
ax2.set_title('Harris win probability')
ax.set_xlabel('Date')
ax.set_ylabel('Electoral Votes')
ax.set_ylim(270-70, 270+70)
# tilt the x-axis labels 45 degrees
ax.set_xticklabels(df['dm'], rotation=45)
ax2.set_xticklabels(df['dm'], rotation=45)

ax.set_yticks(np.arange(210, 331, 20))
ax2.set_ylim(0, 100)
ax2.plot(df[df['source']=='ev']['dm'], df[df['source']=='ev']['demprob'], color='blue', linestyle='-',)
ax2.plot(df[df['source']=='ns']['dm'], df[df['source']=='ns']['demprob'], color='blue', linestyle=':', )
fig.legend(bbox_to_anchor=(.6,.4))

# horizontal line ties
ax.axhline(y=270, color='r', linestyle='--', xmin=0.04, xmax=0.97)
ax2.axhline(y=50, color='r', linestyle='--', xmin=0.04, xmax=0.97)

plt.show()

# save figure
fig.savefig(figdir+'harrisvotes.png', bbox_inches='tight')

# %%
