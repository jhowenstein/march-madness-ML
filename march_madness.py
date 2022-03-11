import numpy as np
import pandas as pd

import os
import glob

import random


def get_win_location_weight(loc):
    if loc == 'H':
        return 0.6
    elif loc == 'A':
        return 1.4
    else:
        return 1

def get_loss_location_weight(loc):
    if loc == 'A':
        return 1.4
    elif loc == 'H':
        return 0.6
    else:
        return 1

def parse_seed(seed_string):
    seed_num = ''
    for letter in seed_string:
        if letter.isdigit():
            seed_num += letter

    seed_int = int(seed_num)
    return seed_int

class Analysis:
    def __init__(self,data_folder='MDataFiles_Stage1'):
        self.data_folder = data_folder
        self.load_data()
        self.seasons = {}

    def load_data(self):
        self.team_data = pd.read_csv(os.path.join(self.data_folder,'MTeams.csv'),index_col=0)
        self.seasons_data = pd.read_csv(os.path.join(self.data_folder,'MSeasons.csv'),index_col=0)
        self.tourney_results = pd.read_csv(os.path.join(self.data_folder,'MNCAATourneyCompactResults.csv'))
        self.tourney_seeds = pd.read_csv(os.path.join(self.data_folder,'MNCAATourneySeeds.csv'))
        self.regular_season_results = pd.read_csv(os.path.join(self.data_folder,'MRegularSeasonCompactResults.csv'))

    def load_seasons(self,start=None,end=None,exclude_2020=True):

        if start is not None and end is not None:
            seasons = self.seasons_data.loc[start:end]
        elif start is not None:
            seasons = self.seasons_data.loc[start:]
        elif end is not None:
            seasons = self.seasons_data.loc[:end]
        else:
            seasons = self.seasons_data
        
        for season in seasons.index:
            if season == 2022:
                continue # Until we have tourney data
            if exclude_2020 and season == 2020:
                continue
            self.load_season(season)

    def load_season(self,year):

        season_data = self.seasons_data.loc[year]
        season_regular_season_results = self.regular_season_results[self.regular_season_results['Season']==year]
        season_tourney_seeds = self.tourney_seeds[self.tourney_seeds['Season']==year].set_index('TeamID')
        season_tourney_results = self.tourney_results[self.tourney_results['Season']==year]

        self.seasons[year] = Season(year,season_data,season_regular_season_results,season_tourney_seeds,season_tourney_results)

    def calc_seasons_features(self):
        for year,season in self.seasons.items():
            print(year)
            season.feature_pipeline()

    def seasons_generate_tourney_model_data(self,seasons,feature_keys=[]):
        features = []
        targets = []
        for season in seasons:
            _X, _y = self.seasons[season].generate_tourney_model_data(feature_keys=feature_keys)
            features.append(_X)
            targets.append(_y)

        X = pd.concat(features,axis=0)
        y = np.concatenate(targets)

        return X, y

    def generate_predictions(seasons=None):
        if seasons is None:
            seasons = np.arange(1985,2022)

    def evaluate_predictions(self,season,predictions_filename):
        predictions = pd.read_csv(predictions_filename,index_col=0)['Pred']

        def score_game(pct):
            score = np.log(pct)
            return score

        season_tourney_seeds = self.tourney_seeds[self.tourney_seeds['Season']==season].set_index('TeamID')
        season_tournament_results = self.tourney_results[self.tourney_results['Season']==season]

        nGames = season_tournament_results.shape[0]

        season_score_total = 0
        for i in season_tournament_results.index:
            game_result = season_tournament_results.loc[i]
            winner_id = game_result['WTeamID']
            loser_id = game_result['LTeamID']

            if winner_id < loser_id:
                pct = predictions[f'{season}_{winner_id}_{loser_id}']
            elif winner_id > loser_id:
                pct = 1 - predictions[f'{season}_{loser_id}_{winner_id}']
                
            season_score_total += score_game(pct)
            
        final_score = - season_score_total / nGames

    def score_model_predictions(self,y,pred):

        logLoss = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / y.shape[0]

        return logLoss

    def train_test_split_seasons(self,test_number=5):
        seasons = list(self.seasons.keys())

        test_seasons = random.sample(seasons,test_number)

        for test_season in test_seasons:
            seasons.remove(test_season)

        return seasons, test_seasons

class Season:
    def __init__(self, year, season_data, regular_season_results, tourney_seeds, tourney_results):
        self.year = year
        self.season_data = season_data
        self.regular_season_results = regular_season_results
        self.tourney_seeds = tourney_seeds
        self.tourney_results = tourney_results

        self.teams = {}

        self.add_teams()

    def add_teams(self):
        for i in self.regular_season_results.index:
            game = self.regular_season_results.loc[i]

            winner_id = game['WTeamID']
            loser_id = game['LTeamID']
            
            if winner_id not in self.teams:
                self.add_team(winner_id)
                
            if loser_id not in self.teams:
                self.add_team(loser_id)
                
            self.teams[winner_id].wins += 1
            if loser_id not in self.teams[winner_id].opponents:
                self.teams[winner_id].opponents[loser_id] = {'wins':1,'losses':0}
            else:
                self.teams[winner_id].opponents[loser_id]['wins'] += 1
                
            self.teams[loser_id].losses += 1
            if winner_id not in self.teams[loser_id].opponents:
                self.teams[loser_id].opponents[winner_id] = {'wins':0,'losses':1}
            else:
                self.teams[loser_id].opponents[winner_id]['losses'] += 1

    def add_team(self,team_id):
        team_win_data = self.regular_season_results[self.regular_season_results['WTeamID']==team_id]
        team_loss_data = self.regular_season_results[self.regular_season_results['LTeamID']==team_id]
        self.teams[team_id] = Team(team_id,team_win_data,team_loss_data)

    def feature_pipeline(self):
        self.calc_teams_win_percentage()
        self.calc_teams_opponents_win_percentage()
        self.calc_teams_oponents_opponents_win_percentage()
        self.calc_teams_win_margin_stats()
        self.assign_tourney_seeds()

    def calc_teams_win_percentage(self):
        for team in self.teams.values():
            team.calc_win_pct()
            team.calc_weighted_win_pct()

    def calc_teams_opponents_win_percentage(self,precision=4):
        for team_id,team in self.teams.items():
            owp_total = 0

            for i in team.win_data.index:
                opp_id = team.win_data.loc[i,'LTeamID']
                h2h_wins = team.opponents[opp_id]['wins']
                h2h_losses = team.opponents[opp_id]['losses']

                opponent = self.teams[opp_id]

                adj_opponent_wins = opponent.wins - h2h_losses
                adj_opponent_games = opponent.games_played - h2h_wins - h2h_losses

                owp = adj_opponent_wins / adj_opponent_games

                owp_total += owp

            for i in team.loss_data.index:
                opp_id = team.loss_data.loc[i,'WTeamID']
                h2h_wins = team.opponents[opp_id]['wins']
                h2h_losses = team.opponents[opp_id]['losses']

                opponent = self.teams[opp_id]

                adj_opponent_wins = opponent.wins - h2h_losses
                adj_opponent_games = opponent.games_played - h2h_wins - h2h_losses

                owp = adj_opponent_wins / adj_opponent_games

                owp_total += owp

            team.features['owp'] = round(owp_total / team.games_played,precision)

    def calc_teams_oponents_opponents_win_percentage(self,precision=4):
        for team_id,team in self.teams.items():
            oowp_total = 0
            for i in team.win_data.index:
                opp_id = team.win_data.loc[i,'LTeamID']
                opp = self.teams[opp_id]
                
                owp_total = 0
                for j in opp.win_data.index:
                    opp_opp_id = opp.win_data.loc[j,'LTeamID']
                    opp_opp = self.teams[opp_opp_id]
                
                    owp_total += opp_opp.win_pct
                    
                for j in opp.loss_data.index:
                    opp_opp_id = opp.loss_data.loc[j,'WTeamID']
                    opp_opp = self.teams[opp_opp_id]
                
                    owp_total += opp_opp.win_pct    
                
                owp = owp_total / opp.games_played
                
                oowp_total += owp
                
            for i in team.loss_data.index:
                opp_id = team.loss_data.loc[i,'WTeamID']
                opp = self.teams[opp_id]
                
                owp_total = 0
                for j in opp.win_data.index:
                    opp_opp_id = opp.win_data.loc[j,'LTeamID']
                    opp_opp = self.teams[opp_opp_id]
                
                    owp_total += opp_opp.win_pct
                    
                for j in opp.loss_data.index:
                    opp_opp_id = opp.loss_data.loc[j,'WTeamID']
                    opp_opp = self.teams[opp_opp_id]
                
                    owp_total += opp_opp.win_pct    
                
                owp = owp_total / opp.games_played
                
                oowp_total += owp
            
            oowp = oowp_total / team.games_played

            team.features['oowp'] = round(oowp,precision)

    def calc_teams_win_margin_stats(self):
        for team_id,team in self.teams.items():
            team.calc_win_margin_stats()

    def assign_tourney_seeds(self):
        for team_id,team in self.teams.items():
            if team_id in self.tourney_seeds.index:
                seed_string = self.tourney_seeds.loc[team_id,'Seed']
                seed_int = parse_seed(seed_string)

                team.features['tourney seed'] = seed_int
            else:
                team.features['tourney seed'] = None

    def generate_tourney_model_data(self,feature_keys=[]):
        model_data = []
        game_results = []

        for index in self.tourney_results.index:
            game = self.tourney_results.loc[index]
            
            winner_id = game['WTeamID']
            loser_id = game['LTeamID']
            
            if winner_id < loser_id:
                team1_id = winner_id
                team2_id = loser_id
                team1_win = 1
            else:
                team1_id = loser_id
                team2_id = winner_id
                team1_win = 0
                
            team1 = self.teams[team1_id]
            team2 = self.teams[team2_id]
                
            team1_features = []
            team2_features = []
            for key in feature_keys:
                team1_features.append(team1.features[key])
                team2_features.append(team2.features[key])
                
            game_features = team1_features + team2_features
            
            model_data.append(game_features)
            game_results.append(team1_win)

        team1_data_names = []
        team2_data_names = []
        for key in feature_keys:
            team1_data_names.append(f'team1 - {key}')
            team2_data_names.append(f'team2 - {key}')
            
        data_names = team1_data_names + team2_data_names

        X = pd.DataFrame(model_data,columns=data_names)
        y = np.array(game_results)

        return X,y


class Team:
    def __init__(self,team_id,team_win_data=None,team_loss_data=None):
        self.id = team_id
        self.win_data = team_win_data
        self.loss_data = team_loss_data

        self.wins = 0
        self.losses = 0

        self.opponents = {}
        self.features = {}

    @property
    def games_played(self):
        return self.wins + self.losses

    @property
    def win_pct(self):
        precision = 3
        pct = self.wins / self.games_played

        return round(pct,precision)

    def calc_win_pct(self):
        self.features['win pct'] = self.win_pct

    def calc_weighted_win_pct(self):
        precision = 3

        weighted_wins = 0
        weighted_losses = 0

        if self.win_data is not None:
            for i in self.win_data.index:
                loc = self.win_data.loc[i]['WLoc']
                
                weighted_wins += get_win_location_weight(loc)
                
        if self.loss_data is not None:
            for i in self.loss_data.index:
                loc = self.loss_data.loc[i]['WLoc']
                
                weighted_losses += get_loss_location_weight(loc)
                
        weighted_win_pct = weighted_wins / (weighted_wins + weighted_losses)

        self.features['weighted win pct'] = round(weighted_win_pct,precision)

    def calc_win_margin_stats(self,precision=2):
        win_margin = self.win_data['WScore'] - self.win_data['LScore']
        loss_margin = self.loss_data['LScore'] - self.loss_data['WScore']

        avg_win_margin = np.mean(win_margin)
        std_win_margin = np.std(win_margin)

        avg_loss_margin = np.mean(loss_margin)
        std_loss_margin = np.std(loss_margin)

        close_wins = sum((win_margin <= 3).astype(int))
        close_losses = sum((loss_margin >= -3).astype(int))

        capped_win_margins = []
        for i in self.win_data.index:
            game = self.win_data.loc[i]
            capped_win_margin = game['WScore'] - game['LScore']
            
            if game['NumOT'] > 0:
                capped_win_margin = 1
            
            if capped_win_margin > 10:
                capped_win_margin = 10
                
            capped_win_margins.append(capped_win_margin)

        capped_avg_win_margin = np.mean(capped_win_margins)
        capped_std_win_margin = np.std(capped_win_margins)

        capped_loss_margins = []
        for i in self.loss_data.index:
            game = self.loss_data.loc[i]
            capped_loss_margin = game['LScore'] - game['WScore']
            
            if game['NumOT'] > 0:
                capped_loss_margin = -1
            
            if capped_loss_margin < -10:
                capped_loss_margin = -10
                
            capped_loss_margins.append(capped_loss_margin)

        capped_avg_loss_margin = np.mean(capped_loss_margins)
        capped_std_loss_margin = np.std(capped_loss_margins)

        if self.loss_data is None:
            avg_loss_margin = 0
            std_loss_margin = 0
            capped_avg_loss_margin = 0
            capped_std_loss_margin = 0

        if self.loss_data.shape[0] <= 1:
            avg_loss_margin = 0
            std_loss_margin = 0
            capped_avg_loss_margin = 0
            capped_std_loss_margin = 0

        self.features['avg win margin'] = round(avg_win_margin,precision)
        self.features['std win margin'] = round(std_win_margin,precision)

        self.features['avg loss margin'] = round(avg_loss_margin,precision)
        self.features['std loss margin'] = round(std_loss_margin,precision)

        self.features['capped avg win margin'] = round(capped_avg_win_margin,precision)
        self.features['capped std win margin'] = round(capped_std_win_margin,precision)

        self.features['capped avg loss margin'] = round(capped_avg_loss_margin,precision)
        self.features['capped std loss margin'] = round(capped_std_loss_margin,precision)

        self.features['close wins'] = close_wins
        self.features['close losses'] = close_losses

class Game:
    def __init__(self):
        pass