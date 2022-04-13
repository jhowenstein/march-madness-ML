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

def bound_predictions(predictions,alpha=0.025):
    bounded_predictions = []
    for pred in predictions:
        if pred > (1-alpha):
            pred = (1-alpha)
        elif pred < alpha:
            pred = alpha
            
        bounded_predictions.append(pred)
        
    return np.array(bounded_predictions)

class Analysis:
    def __init__(self,data_folder='MDataFiles_Stage2'):
        self.data_folder = data_folder
        self.load_data()
        self.seasons = {}

    def load_data(self):
        mf_character = self.data_folder[0]

        self.team_data = pd.read_csv(os.path.join(self.data_folder,f'{mf_character}Teams.csv'),index_col=0)
        self.seasons_data = pd.read_csv(os.path.join(self.data_folder,f'{mf_character}Seasons.csv'),index_col=0)
        self.tourney_results = pd.read_csv(os.path.join(self.data_folder,f'{mf_character}NCAATourneyCompactResults.csv'))
        self.tourney_seeds = pd.read_csv(os.path.join(self.data_folder,f'{mf_character}NCAATourneySeeds.csv'))
        self.regular_season_results = pd.read_csv(os.path.join(self.data_folder,f'{mf_character}RegularSeasonCompactResults.csv'))

        self.process_input_data()

    def process_input_data(self):

        seeds = self.tourney_seeds['Seed'].values
        self.tourney_seeds['Numerical Seed'] = [parse_seed(seed) for seed in seeds]

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
            if exclude_2020 and season == 2020:
                continue
            self.load_season(season)

    def load_season(self,year):

        season_data = self.seasons_data.loc[year]
        season_regular_season_results = self.regular_season_results[self.regular_season_results['Season']==year]
        season_tourney_seeds = self.tourney_seeds[self.tourney_seeds['Season']==year].set_index('TeamID')
        season_tourney_results = self.tourney_results[self.tourney_results['Season']==year]

        self.seasons[year] = Season(year,season_data,season_regular_season_results,season_tourney_seeds,season_tourney_results)

    def calc_seasons_features(self,detailed_stats_features=True):
        for year,season in self.seasons.items():
            print(year)
            season.feature_pipeline(detailed_stats_features=detailed_stats_features)

    def seasons_generate_tourney_model_data(self,feature_keys=[],seasons=None,fill_nan=True):
        if seasons is None:
            seasons = list(self.seasons.keys())[:-1]

        features = []
        targets = []
        for season in seasons:
            if season == 2020:
                continue
            
            _X, _y = self.seasons[season].generate_tourney_model_data(feature_keys=feature_keys)

            features.append(_X)
            targets.append(_y)

        X = pd.concat(features,axis=0)
        y = np.concatenate(targets)

        if fill_nan:
            for key in X.keys():
                col = X[key]
                X[key] = col.fillna(col.median())

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

        return final_score

    def score_model_predictions(self,y,pred):

        logLoss = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / y.shape[0]

        return logLoss

    def train_test_split_seasons(self,test_number=5):
        seasons = list(self.seasons.keys())

        test_seasons = random.sample(seasons,test_number)

        for test_season in test_seasons:
            seasons.remove(test_season)

        return seasons, test_seasons

    def generate_seed_win_predictions(self,df):
        pcts = pd.read_csv('seed win percentage.csv')

        df.index = np.arange(df.shape[0])
        preds = []
        for i in df.index:
            game = df.loc[i]
            team1_seed = game['team1 - tourney seed']
            team2_seed = game['team2 - tourney seed']

            if team1_seed == team2_seed:
                pred = 0.5
            elif team1_seed < team2_seed:
                pred = pcts[(pcts['Team Seed']==team1_seed)&(pcts['Opponent Seed']==team2_seed)]['pct'].values[0]
            elif team2_seed < team1_seed:
                pred = 1 - pcts[(pcts['Team Seed']==team2_seed)&(pcts['Opponent Seed']==team1_seed)]['pct'].values[0]
                
            preds.append(pred)
            
        preds = np.array(preds)

        return preds

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

    def feature_pipeline(self,detailed_stats_features=True):
        self.calc_teams_win_percentage()
        self.calc_teams_opponents_win_percentage()
        self.calc_teams_oponents_opponents_win_percentage()
        self.calc_teams_win_margin_stats()
        self.assign_tourney_seeds()
        self.calc_quality_wins()
        self.calc_teams_late_season_form()
        if detailed_stats_features:
            # These features are fairly time intensive so they can be skipped if unused
            self.calc_teams_detailed_game_stats()
            self.calc_teams_avg_detailed_game_stats()
            self.calc_teams_net_detailed_game_stats()

    def calc_teams_win_percentage(self):
        for team in self.teams.values():
            team.calc_win_pct()
            team.calc_weighted_win_pct()

    def calc_quality_wins(self):

        top64 = self.tourney_seeds
        top32 = self.tourney_seeds[self.tourney_seeds['Numerical Seed'] <= 8]
        top16 = self.tourney_seeds[self.tourney_seeds['Numerical Seed'] <= 4]
        top8 = self.tourney_seeds[self.tourney_seeds['Numerical Seed'] <= 2]

        for team_id,team in self.teams.items():

            top64_wins = team.win_data.merge(top64,left_on='LTeamID',right_index=True)
            top32_wins = team.win_data.merge(top32,left_on='LTeamID',right_index=True)
            top16_wins = team.win_data.merge(top16,left_on='LTeamID',right_index=True)
            top8_wins = team.win_data.merge(top8,left_on='LTeamID',right_index=True)

            top64_losses = team.loss_data.merge(top64,left_on='WTeamID',right_index=True)
            top32_losses = team.loss_data.merge(top32,left_on='WTeamID',right_index=True)
            top16_losses = team.loss_data.merge(top16,left_on='WTeamID',right_index=True)
            top8_losses = team.loss_data.merge(top8,left_on='WTeamID',right_index=True)

            weighted_top64_wins = 0
            weighted_top32_wins = 0
            weighted_top16_wins = 0
            weighted_top8_wins = 0

            weighted_top64_losses = 0
            weighted_top32_losses = 0
            weighted_top16_losses = 0
            weighted_top8_losses = 0

            for i in top64_wins.index:
                loc = top64_wins.loc[i]['WLoc']
                weighted_top64_wins += get_win_location_weight(loc)

            for i in top32_wins.index:
                loc = top32_wins.loc[i]['WLoc']
                weighted_top32_wins += get_win_location_weight(loc)

            for i in top16_wins.index:
                loc = top16_wins.loc[i]['WLoc']
                weighted_top16_wins += get_win_location_weight(loc)

            for i in top8_wins.index:
                loc = top8_wins.loc[i]['WLoc']
                weighted_top8_wins += get_win_location_weight(loc)
                    
            for i in top64_losses.index:
                loc = top64_losses.loc[i]['WLoc']           
                weighted_top64_losses += get_loss_location_weight(loc)

            for i in top32_losses.index:
                loc = top32_losses.loc[i]['WLoc']           
                weighted_top32_losses += get_loss_location_weight(loc)

            for i in top16_losses.index:
                loc = top16_losses.loc[i]['WLoc']           
                weighted_top16_losses += get_loss_location_weight(loc)

            for i in top8_losses.index:
                loc = top8_losses.loc[i]['WLoc']           
                weighted_top8_losses += get_loss_location_weight(loc)

            team.features['weighted top64 wins'] = weighted_top64_wins
            team.features['weighted top32 wins'] = weighted_top32_wins
            team.features['weighted top16 wins'] = weighted_top16_wins
            team.features['weighted top8 wins'] = weighted_top8_wins

            team.features['weighted top64 losses'] = weighted_top64_losses
            team.features['weighted top32 losses'] = weighted_top32_losses
            team.features['weighted top16 losses'] = weighted_top16_losses
            team.features['weighted top8 losses'] = weighted_top8_losses

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

    def calc_teams_net_detailed_game_stats(self,precision=4):
        metric_keys = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO%','Stl%','Blk%','PF',
                       'TR','FGM2','FGA2','FG%','FG2%','FG3%','FGA3%','FT%','Pos','OEff']

        for team_id,team in self.teams.items():
            for metric in metric_keys:
                net_values = []
                game_count = 0
                for i in team.win_data.index:
                    game = team.win_data.loc[i]
                    
                    team_game_value = game[f'W{metric}']
                    
                    opp_id = game['LTeamID']
                    opp_avg_value = self.teams[opp_id].features[f'Opp Avg {metric}']
                    
                    if np.isnan(team_game_value) or np.isnan(opp_avg_value):
                        continue
                    
                    net_value = team_game_value - opp_avg_value
                    
                    net_values.append(net_value)
                    
                for i in team.loss_data.index:
                    game = team.loss_data.loc[i]
                    
                    team_game_value = game[f'L{metric}']
                    
                    opp_id = game['WTeamID']
                    opp_avg_value = self.teams[opp_id].features[f'Opp Avg {metric}']
                    
                    if np.isnan(team_game_value) or np.isnan(opp_avg_value):
                        continue
                    
                    net_value = team_game_value - opp_avg_value
                    
                    net_values.append(net_value)
        
                team_net_value = np.mean(net_values)

                if np.isnan(team_net_value):
                    team_net_value = 0

                team.features[f'Net Team Avg {metric}'] = round(team_net_value,precision)

            for metric in metric_keys:
                net_values = []
                game_count = 0
                for i in team.win_data.index:
                    game = team.win_data.loc[i]
                    
                    opp_game_value = game[f'L{metric}']
                    
                    opp_id = game['LTeamID']
                    opp_avg_value = self.teams[opp_id].features[f'Team Avg {metric}']
                    
                    if np.isnan(opp_game_value) or np.isnan(opp_avg_value):
                        continue
                    
                    net_value = opp_game_value - opp_avg_value
                    
                    net_values.append(net_value)
                    
                for i in team.loss_data.index:
                    game = team.loss_data.loc[i]
                    
                    opp_game_value = game[f'W{metric}']
                    
                    opp_id = game['WTeamID']
                    opp_avg_value = self.teams[opp_id].features[f'Team Avg {metric}']
                    
                    if np.isnan(opp_game_value) or np.isnan(opp_avg_value):
                        continue
                    
                    net_value = opp_game_value - opp_avg_value
                    
                    net_values.append(net_value)
                    
                team_net_value = np.mean(net_values)

                if np.isnan(team_net_value):
                    team_net_value = 0

                team.features[f'Net Opp Avg {metric}'] = round(team_net_value,precision)

    def calc_teams_win_margin_stats(self):
        for team_id,team in self.teams.items():
            team.calc_win_margin_stats()

    def calc_teams_late_season_form(self):
        for team_id,team in self.teams.items():
            team.calc_late_season_form()

    def calc_teams_detailed_game_stats(self):
        for team_id,team in self.teams.items():
            team.calc_detailed_game_stats()

    def calc_teams_avg_detailed_game_stats(self):
        for team_id,team in self.teams.items():
            team.calc_avg_detailed_game_stats()

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

    def calc_late_season_form(self,precision=3):
        season_data = pd.concat([self.win_data,self.loss_data]).sort_values('DayNum')

        last10_data = season_data[-10:]
        last5_data = season_data[-5:]

        last10_win_count = 0
        last10_weighted_win_count = 0
        last10_weighted_loss_count = 0

        for i in last10_data.index:
            game = last10_data.loc[i]
            
            winner_id = game['WTeamID']
            winner_loc = game['WLoc']
            
            if winner_id == self.id:
                last10_win_count += 1
                last10_weighted_win_count += get_win_location_weight(winner_loc)
            else:
                last10_weighted_loss_count += get_loss_location_weight(winner_loc)

        last10_win_pct = last10_win_count / last10_data.shape[0]
        last10_weighted_win_pct = round(last10_weighted_win_count / (last10_weighted_win_count + last10_weighted_loss_count),precision)

        last5_win_count = 0
        last5_weighted_win_count = 0
        last5_weighted_loss_count = 0

        for i in last5_data.index:
            game = last5_data.loc[i]
            
            winner_id = game['WTeamID']
            winner_loc = game['WLoc']
            
            if winner_id == self.id:
                last5_win_count += 1
                last5_weighted_win_count += get_win_location_weight(winner_loc)
            else:
                last5_weighted_loss_count += get_loss_location_weight(winner_loc)
                
        last5_win_pct = last5_win_count / last5_data.shape[0]
        last5_weighted_win_pct = round(last5_weighted_win_count / (last5_weighted_win_count + last5_weighted_loss_count),precision)

        conf_tourney_wins = 0

        for i in last10_data.index[::-1]:
            game = last10_data.loc[i]
            
            loc = game['WLoc']
            
            if loc != 'N':
                break
            else:
                if game['WTeamID'] == self.id:
                    conf_tourney_wins += 1

        last_game_index = last5_data.index[-1]
        last_game = last5_data.loc[last_game_index]
        if last_game['WTeamID'] == self.id:
            conf_champ = 1
        else:
            conf_champ = 0

        self.features['last10 win pct'] = last10_win_pct
        self.features['last10 weighted win pct'] = last10_weighted_win_pct

        self.features['last5 win pct'] = last5_win_pct
        self.features['last5 weighted win pct'] = last5_weighted_win_pct

        self.features['conference tourney wins'] = conf_tourney_wins
        self.features['conference champ'] = conf_champ

    def calc_detailed_game_stats(self):
        for df in (self.win_data,self.loss_data):
            for t in ('W','L'):
                df[f'{t}TR'] = df[f'{t}OR'] + df[f'{t}DR']
                df[f'{t}FGM2'] = df[f'{t}FGM'] - df[f'{t}FGM3']
                df[f'{t}FGA2'] = df[f'{t}FGA'] - df[f'{t}FGA3']
                df[f'{t}FG%'] = df[f'{t}FGM'] / df[f'{t}FGA']
                df[f'{t}FG2%'] = df[f'{t}FGM2'] / df[f'{t}FGA2']
                df[f'{t}FG3%'] = df[f'{t}FGM3'] / df[f'{t}FGA3']
                df[f'{t}FGA3%'] = df[f'{t}FGA3'] / df[f'{t}FGA']
                df[f'{t}FT%'] = df[f'{t}FTM'] / df[f'{t}FTA']
                df[f'{t}Pos'] = (0.96 * df[f'{t}FGA']) - df[f'{t}OR'] + df[f'{t}TO'] + (0.44 * df[f'{t}FTA'])
                df[f'{t}TO%'] = df[f'{t}TO'] / df[f'{t}Pos']
                df[f'{t}Stl%'] = df[f'{t}Stl'] / df[f'{t}Pos']
                df[f'{t}Blk%'] = df[f'{t}Blk'] / df[f'{t}Pos']
                df[f'{t}FT%'] = df[f'{t}FTM'] / df[f'{t}FTA']
                df[f'{t}OEff'] = df[f'{t}Score'] / df[f'{t}Pos']

    def calc_avg_detailed_game_stats(self,precision=4):
        metric_keys = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO%','Stl%','Blk%','PF',
                       'TR','FGM2','FGA2','FG%','FG2%','FG3%','FGA3%','FT%','Pos','OEff']

        for metric in metric_keys:
            team_win_values = self.win_data[f'W{metric}'].values
            team_loss_values = self.loss_data[f'L{metric}'].values
            
            team_values = np.concatenate((team_win_values,team_loss_values))
            team_avg = team_values.mean()
            
            self.features[f'Team Avg {metric}'] = round(team_avg,precision)
            
        for metric in metric_keys:
            team_win_values = self.win_data[f'L{metric}'].values
            team_loss_values = self.loss_data[f'W{metric}'].values
            
            team_values = np.concatenate((team_win_values,team_loss_values))
            team_avg = team_values.mean()
            
            self.features[f'Opp Avg {metric}'] = round(team_avg,precision)

class Game:
    def __init__(self):
        pass
