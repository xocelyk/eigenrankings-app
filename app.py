from re import L
import streamlit as st
import pandas as pd
import numpy as np
from sportsreference.ncaab.teams import Teams
from scipy.sparse.linalg import eigs
from scipy.stats import norm
from matplotlib import pyplot as plt

@st.cache
def get_teams():
    return Teams()
teams = get_teams()

hca = st.sidebar.checkbox('Include home-court advantage')
HCA_VAL = 1.25

st.sidebar.header('Change scoring function')

# Let user change coefficient for logistic regression
K = st.sidebar.slider('Choose k.', min_value=0, max_value=100, value=10)
margin_f = lambda margin: 1/ (1 + np.exp(-K/100 * margin))

# Show (plot) result
x_sample =[i* .1 for i in range(-500, 500)]
y_sample = [margin_f(x) for x in x_sample]
plt.style.use('ggplot')
fig, ax = plt.subplots() 
ax.plot(x_sample, y_sample)
# make y axis between 0 and 1
ax.set_ylim(0, 1)
ax.set_xlabel('margin')
ax.set_ylabel('value rewarded (edge weight)')
ax.set_title('reward = 1 / (1 + e^(-k/100 * margin))')
st.sidebar.pyplot(fig)

def get_percentile_rankings(rankings):
    std = np.std([i[1] for i in rankings.values()])
    mean = np.mean([i[1] for i in rankings.values()])
    res = {}
    for key, value in rankings.items():
        res[key] = [value[0], round(100 * norm.cdf((value[1] - mean)/std), 2), value[1]]
    return res

def get_teams_games_played(data_dict, teams_list):
    res = {team: 0 for team in teams_list}
    for key, game in data_dict.items():
        if game[2] != 'None':
            res[game[0]] += 1
            res[game[1]] += 1
    return res

@st.cache
def get_teams_list():
    teams_list = [team.name for team in teams]
    return teams_list
                
@st.cache
def load_data(teams_list):
    data_dict = {}
    games_tracked = []
    counter = 0
    for team in Teams():
        for game in team.schedule:
            # if game.boxscore_index not in games_tracked:
            games_tracked.append(game.boxscore_index)
            if game.opponent_name not in teams_list:
                continue
            game_data = [team.name, game.opponent_name, game.points_for, game.points_against, game.location, game.date, game.datetime]
            # print(game_data)
            bs_index = game.boxscore_index if game.boxscore_index is not None else counter
            counter += 1
            data_dict[bs_index] = game_data
    return data_dict

def get_adj_matrix(teams_list, data_dict):
    games_played_dict = {team: 0 for team in teams_list}
    for key, value in data_dict.items():
        if value[2] is not None:
            games_played_dict[value[0]] += 1
            games_played_dict[value[1]] += 1

    # data_dict has key index and value [team1, team2, team1 score, team2 score, location]
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['team1', 'team2', 'team1 score', 'team2 score', 'location', 'date', 'datetime'])
    df = df.dropna()
    teams_idx = {}
    idx = 0
    for team in teams_list:
        teams_idx[team] = idx
        idx += 1
    
    adj_matrix = pd.DataFrame(0, index=range(len(teams_list)), columns=range(len(teams_list)))

    for team in teams_list:
        team_df = df[df['team1'] == team]
        row_idx = teams_idx[team]
        for index, row in team_df.iterrows():
            if row['team1'] in teams_idx:
                col_idx = teams_idx[row['team2']]
                margin = row['team1 score'] - row['team2 score']
                if hca:
                    if row['location'] == 'Home':
                        margin += HCA_VAL
                    elif row['location'] == 'Away':
                        margin -= HCA_VAL
                adj_matrix.iloc[row_idx, col_idx] += margin_f(margin)
                adj_matrix.iloc[col_idx, row_idx] += margin_f(-margin)
    return adj_matrix

@st.cache
def load_future_games(data_dict):
    future_games = {}
    games_collected = []
    for key, value in data_dict.items():
        if value[2] is None:
            if [value[1], value[0], value[3]] in games_collected:
                continue            # team1, team2, location, date
            future_games[key] = [value[0], value[1], value[-2], value[-1]]
            games_collected.append([value[0], value[1], value[3]])
    return future_games

def turn_to_pctl(ranked):
    res = {}
    st_dev = np.std([i[1] for i in ranked.values()])
    print(st_dev)
    mean = np.mean([i[1] for i in ranked.values()])
    print(mean)
    for key, value in ranked.items():
        res[key] = (value[0], round(100 * norm.cdf(value[1] - mean/st_dev), 2))
    return res

def get_rankings(adj_matrix, teams_list, data_dict):
# normalize the matrix    
    adj_matrix = adj_matrix.to_numpy()
    temp_matrix = np.transpose(adj_matrix)
    for i in range(len(temp_matrix)):
        temp_matrix[i] = temp_matrix[i]/sum(temp_matrix[i])
        if i == 3 or i == 4:
            print(temp_matrix[i])
            print(sum(temp_matrix[i]))
    
    teams_games_played = get_teams_games_played(data_dict, teams_list)
    for i in range(len(adj_matrix)):
        adj_matrix[i] = adj_matrix[i]/teams_games_played[teams_list[i]]

    adj_matrix = np.transpose(temp_matrix)

    # now adjust for the games played by each team

    # take the eigenvector of the transpose of matrix adj_matrix
    val, vec = eigs(adj_matrix, which='LM', k=1)
    vec = np.ndarray.flatten(abs(vec))
    sorted_indices = vec.argsort()
    # This is just arbitrary to turn the rankings nicer looking
    ranked = [(teams_list[i], round(100*(vec[i])**.2, 2)) for i in sorted_indices]
    ranked.reverse()
    ranked = {i + 1 : ranked[i] for i in range(len(ranked))}
    return ranked

def main():
    teams_list = get_teams_list()
    data = load_data(teams_list)
    teams_games_played = get_teams_games_played(data, teams_list)
    adj_matrix = get_adj_matrix(teams_list, data)
    future_games = load_future_games(data)
    future_games = pd.DataFrame.from_dict(future_games, orient='index', columns=['team1', 'team2', 'location', 'date'])
    chosen_teams = st.sidebar.multiselect('Select the teams whose games you want to edit', options=teams_list)
    if len(chosen_teams) > 0:
        games_to_edit = future_games[future_games['team1'].isin(chosen_teams) | future_games['team2'].isin(chosen_teams)]
        # sort games_to_edit by date
        sort_val_f = lambda date_string: str(date_string).split(' ')[0].split('-')[0] * 31**2 + str(date_string).split(' ')[0].split('-')[1] * 31 + str(date_string).split(' ')[0].split('-')[2]
        # create new columns in dataframe from applying sort_val_f to date

        games_to_edit['sort_val'] = games_to_edit['date'].apply(sort_val_f)
        games_to_edit = games_to_edit.sort_values(by='sort_val')
        for boxscore_idx, game in games_to_edit.iterrows():
            home_team = game['team1'] if game['location'] == 'Home' else game['team2']
            winner = st.sidebar.radio('Choose winner: ' + str(str(game['date']).split(' ')[0]) + ' Home: ' + home_team, options=['None'] + [game['team1'], game['team2']], key=boxscore_idx)
            if winner != 'None':
                margin = st.sidebar.slider('Margin of victory', min_value=0, max_value=100, step=1, value=10, key=boxscore_idx)
                if hca:
                    if game['location'] == 'Home':
                        margin += HCA_VAL
                    elif game['location'] == 'Away':
                        margin -= HCA_VAL
                if winner == game['team1']:
                    pass
                elif winner == game['team2']:
                    margin = -margin
                row_idx = teams_list.index(game['team1'])
                col_idx = teams_list.index(game['team2'])
                adj_matrix.iloc[row_idx, col_idx] += margin_f(margin)
                adj_matrix.iloc[col_idx, row_idx] += margin_f(-margin)
    rankings = get_rankings(adj_matrix, teams_list, data)
    rankings = get_percentile_rankings(rankings)

    # change to str because of streamlit bug
    df = pd.DataFrame.from_dict(rankings, orient='index', columns = ['team', 'population adjusted rating', 'eigenvector rating']).astype(str)
    st.dataframe(df)

    st.header('Motivation')
    st.write('The idea behind all of this is an old one. I\'m using PageRank to rank college basketball teams. I encourage you to Google PageRank if you haven\'t heard of it. \
             You can imagine a graph of where each node is a team. The edges represent the games that have been played. We want to determine the value of those edges. \
            The edge pointing from the winner to the loser should have a greater weight than the edge pointing from the loser to the winner.')
    st.write('I use a the margin of the game to inform the edge weights. A negative margin (you lost) is should yield a smaller edge weight than a positive margin (you won). \
            But all our values of the edge weights need to be positive, both for the math to work and for the graph to make real sense. For the time being, I\'d like to suppose each game is weighted equally \
            (though you can imagine other approaches to this: perhaps more recent games have higher weight, games with higher partiy have higher weight, etc.). \
            So we want a function that both maps margin (which ranges from (-inf, +inf) to some bounded non-negative interval. We also want for this function f to be such that there exists an e such that for all margins in the reals there exists an e such that f(margin) + f(-margin) = e.  \
            (This is the criterion that satsifies the weights of all games being equal.)')
    st.write('I use a logistic function of the margin of victory to determine the points alloted to any team. \
            The function is value = 1 / (1 + np.exp(-k/100 * margin)). There are a couple nice things about this function. The first is that \
            it f(margin) + f(-margin) = 1, so for all games 1 point is rewarded. Another thing is that you get diminishing returns for blowouts. The idea is that \
            the difference between a 30 point win and 40 point win is less substantial than the difference between a 10 point win and a 20 point win. \
            Another way to think about it is that the absolute value of the function is always concave down. \n \
            I\'m leaving it open to the user to mess around with a value of k to see what works best. In my own rankings I manually adjust the value of k while paying attention to the tradeoff between wins predicted rate and wins explained rate (I call these predictive / retrodictive accuracy). \
            There\'s certainly a better way to do it, and maybe it just requires me being more rigorous about my current method. But I thought it would be fun to make a toy where you can fiddle with this logistic function constant k and see what different rankings it yields.') 
    # Introduce idea of editing future games
    st.write('Another thing I wanted to allow people to do was test out future games. I thought about listing all future games but there are too many to list nicely. \
            Unfortunately, too, I could not figure out a way to display a sort of matrix of future games, or do anything graphically interesting. \
            What I decided was the best solution was to allow the user to choose a team, and then see that team\'s future games, and edit those. ')      

if __name__ == '__main__':
    main()

